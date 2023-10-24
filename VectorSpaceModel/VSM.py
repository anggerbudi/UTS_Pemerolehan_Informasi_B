"""
Kelas       : Pemerolehan Informasi B
Tugas       : VectorSpaceModel Tugas UTS
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
            # print(word + " " + str(idf))
            while linkedlist.next_node is not None:
                linkedlist = linkedlist.next_node
                linkedlist.tfidf = idf * linkedlist.freq
                # print(linkedlist.freq)
                # print(" " + str(linkedlist.tfidf))
            # print("\n")

    @staticmethod
    def vec_d_matrix(unique_words_all, files_with_index, linked_list_data):
        dict_words = list(unique_words_all)
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

    def output(self, vec_d, vec_q):
        d_cosines = [self.cosine_sim(vec_q, d) for d in vec_d]
        # for i, cosine in enumerate(d_cosines, start=1):
        #     if cosine is not None:
        #         print(files_with_index[i] + ' ' + str(cosine))
        return d_cosines

    def vector_relevance(self, d_cosines, files_with_index):
        relevance_doc = ['sains2.txt', 'sains3.txt', 'sains6.txt', 'sains20.txt', 'sains54.txt', 'sains72.txt',
                         'sains85.txt', 'sains89.txt', 'sains98.txt', 'sains104.txt', 'sains105.txt', 'sains114.txt']
        relevance_vector = [1 if cosine_value > 0 else 0 for cosine_value in d_cosines]
        ground_truth = [1 if file_name in relevance_doc else 0 for file_name in files_with_index.values()]
        print(relevance_vector)
        print(ground_truth)
        return relevance_vector, ground_truth

    def hitung_recall_precision(self, predicted_val, true_val):
        true_positive = sum(
            [1 for a, b in zip(predicted_val, true_val) if a == b and a == 1])
        false_positive = sum(
            [1 for a, b in zip(predicted_val, true_val) if a == 0 and b == 1])
        false_negative = sum(
            [1 for a, b in zip(predicted_val, true_val) if a == 1 and b == 0])

        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative != 0 else 0
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive != 0 else 0

        print('recall value : ', recall)
        print('precision value : ', precision)
        return recall, precision
    
    def hitung_rochio(self, d_cosines, files_with_index):
        relevance_doc = ['sains2.txt', 'sains3.txt', 'sains6.txt', 'sains20.txt', 'sains54.txt', 'sains72.txt',
                         'sains85.txt', 'sains89.txt', 'sains98.txt', 'sains104.txt', 'sains105.txt', 'sains114.txt']
        relevant_cosine_values = []
        non_relevant_cosine_values = []

        # Iterate through the list of all documents and their cosine similarities
        for doc, cosine_value in zip(files_with_index.values(), d_cosines):
            if doc in relevance_doc:
                relevant_cosine_values.append(cosine_value)
            else:
                non_relevant_cosine_values.append(cosine_value)
        import numpy as np

        # Initialize the query vector for "fenomena antariksa"
        original_query_vector = d_cosines  # Replace with the actual vector

        # Define the alpha, beta, and gamma values
        alpha = 1
        beta = 0.75
        gamma = 0.25

        # Relevant and non-relevant document vectors (average vectors)
        relevant_document_vectors = relevant_cosine_values  # Replace with actual vectors
        non_relevant_document_vectors = non_relevant_cosine_values  # Replace with actual vectors

        # Calculate the new query vector using Rocchio formula
        new_query_vector = alpha * original_query_vector + beta * np.mean(relevant_document_vectors, axis=0) - gamma * np.mean(non_relevant_document_vectors, axis=0)

        # Normalize the new query vector to have unit length
        new_query_vector /= np.linalg.norm(new_query_vector)
        print('asd',new_query_vector)
        return new_query_vector
        # Now, you can use the new_query_vector to re-rank and retrieve documents in your collection
        # Compute cosine similarity between the new_query_vector and documents to rank them

        # Re-rank your documents based on cosine similarity with the new query vector

        # Evaluate the results using precision and recall

