"""
Kelas       : Pemerolehan Informasi B
Tugas       : Program Tugas UTS
Kelompok    : Martinus Angger Budi Wicaksono    (215314163)
              Antonius Yoga Chris Raharja       (215314175)
              Antonius Miquel Aureliano         (215314181)
"""

import glob
import math
import os.path
import re

import nltk
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from DataStructure import NodeClass, LinkedListClass


class Program:
    def __init__(self, file_path):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.file_folder = file_path
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = StemmerFactory().create_stemmer()

    @staticmethod
    def find_freq_and_unique_word(words):
        words_unique = set(word for word in words)
        word_freq = {word: words.count(word) for word in words_unique}
        return word_freq

    @staticmethod
    def remove_special_char(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    def text_loader_and_words_collector(self):
        files_with_index = {}
        unique_words_all = set()

        for idx, doc in enumerate(glob.glob(self.file_folder), start=1):
            with open(doc, 'r') as file:
                text = re.sub(r'\d', '', self.remove_special_char(file.read()))
                words = [self.stemmer.stem(word.lower()) for word in word_tokenize(text)
                         if len(word) >= 1 and word.lower() not in self.stop_words]
                dict_global = self.find_freq_and_unique_word(words)
                files_with_index[idx] = os.path.basename(doc)
                unique_words_all.update(dict_global.keys())
                print('ini dict global : ', dict_global)
        print('ini unique words all : ', unique_words_all)
        return unique_words_all, files_with_index

    def process_data_and_update_linked_list(self, unique_words_all):
        linked_list_data = {}
        for word in unique_words_all:
            linked_list_data[word] = LinkedListClass()
            linked_list_data[word].head = NodeClass(1, NodeClass)
        for idx, doc in enumerate(glob.glob(self.file_folder), start=0):
            with open(doc, 'r') as file:
                text = re.sub(r'\d', '', self.remove_special_char(file.read()))
                words = [self.stemmer.stem(word.lower()) for word in word_tokenize(text)
                         if len(word) >= 1 and word.lower() not in self.stop_words]
                word_freq_in_doc = self.find_freq_and_unique_word(words)
                for word in word_freq_in_doc.keys():
                    linked_list = linked_list_data[word].head
                    linked_list_data[word].num_docs = linked_list_data[word].num_docs + 1
                    while linked_list.next_node is not None:
                        linked_list = linked_list.next_node
                    linked_list.next_node = NodeClass(idx, word_freq_in_doc[word])
        return linked_list_data

    @staticmethod
    def calculate_tfidf_for_words(unique_words_all, linked_list_data, files_with_index):
        n_tot = len(files_with_index)
        for word in unique_words_all:
            linkedlist = linked_list_data[word].head
            df = linked_list_data[word].num_docs
            idf = math.log2(n_tot / df) + 1
            linked_list_data[word].idf = idf
            print(word + " " + str(idf))
            while linkedlist.next_node is not None:
                linkedlist = linkedlist.next_node
                linkedlist.tfidf = idf * linkedlist.freq
                print(linkedlist.freq)
                print(" " + str(linkedlist.tfidf))
            print("\n")

    @staticmethod
    def vec_d_matrix(unique_words_all, files_with_index, linked_list_data):
        dict_words = list(unique_words_all)
        print('ini dict words : ', dict_words)
        total_files = len(files_with_index)
        total_vocab = len(dict_words)
        vec_d = np.zeros((total_files, total_vocab))
        for i in range(len(dict_words)):
            linkedlist = linked_list_data[dict_words[i]].head
            while linkedlist.next_node is not None:
                linkedlist = linkedlist.next_node
                vec_d[linkedlist.doc][i] = linkedlist.tfidf
        return dict_words, total_files, total_vocab, vec_d

    def query_input(self, total_vocab, dict_words, linked_list_data):
        query = input('Enter your query: ')
        wordsq = [self.stemmer.stem(word.lower()) for word in word_tokenize(query)
                  if len(word) >= 1 and word.lower() not in self.stop_words]
        dict_query = {word: self.find_freq_and_unique_word([word])[word] for word in wordsq}
        vec_q = np.zeros(total_vocab)
        for word, freq in dict_query.items():
            index = dict_words.index(word)
            vec_q[index] = linked_list_data[word].idf * freq
        return vec_q

    @staticmethod
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def output(self, vec_d, vec_q, total_files, files_with_index):
        d_cosines = [self.cosine_sim(vec_q, d) for d in vec_d]
        ranked_indices = np.argsort(d_cosines)[-total_files:][::-1]
        print(ranked_indices)
        for idx in ranked_indices:
            print(files_with_index[idx + 1])
