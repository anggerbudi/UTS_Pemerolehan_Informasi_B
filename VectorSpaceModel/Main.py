"""
Kelas       : Pemerolehan Informasi B
Tugas       : VectorSpaceModel Tugas UTS
Kelompok    : Martinus Angger Budi Wicaksono    (215314163)
              Antonius Yoga Chris Raharja       (215314175)
              Antonius Miquel Aureliano         (215314181)
"""


from VectorSpaceModel import VSM

if __name__ == "__main__":
    file_folder = "C:/Users/angger/PycharmProjects/UTS_Pemerolehan Informasi B/dok_soal/*"
    main = VSM.Program(file_folder)

    unique_words_all, files_with_index = main.text_loader_and_words_collector()
    linked_list_data = main.process_data_and_update_linked_list(unique_words_all)
    main.calculate_tfidf_for_words(unique_words_all, linked_list_data, files_with_index)
    dict_words, total_files, total_vocab, vecD = main.vec_d_matrix(unique_words_all, files_with_index, linked_list_data)

    vecQ = main.query_input(total_vocab, dict_words, linked_list_data)
    d_cosines = main.output(vecD, vecQ, total_files, files_with_index)
    predicted_relevance, true_relevance = main.vector_relevance(d_cosines, files_with_index)
    recall, precision = main.hitung_recall_precision(predicted_relevance, true_relevance)
    rochio = main.hitung_rochio(d_cosines, files_with_index)
    predicted_relevance2, true_relevance = main.vector_relevance(rochio, files_with_index)
    recall2, precision2 = main.hitung_recall_precision(predicted_relevance, true_relevance)
    print('asd',recall2)
    print('dsa',precision2)
    