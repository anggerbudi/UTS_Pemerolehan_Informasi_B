"""
Kelas       : Pemerolehan Informasi B
Tugas       : VectorSpaceModel Tugas UTS
Kelompok    : Martinus Angger Budi Wicaksono    (215314163)
              Antonius Yoga Chris Raharja       (215314175)
              Antonius Miquel Aureliano         (215314181)
"""


from VectorSpaceModel import VSM

if __name__ == "__main__":
    file_folder = "C:/Users/anton/PycharmProjects/UTS_Pemerolehan_Informasi_B/dok_soal/*"
    main = VSM.Program(file_folder)

    print("No 2 : Load file dan preprocessing")
    unique_words_all, files_with_index = main.text_loader_and_words_collector()
    linked_list_data = main.process_data_and_update_linked_list(unique_words_all)
    #TODO : print one of the doc

    print("No 3 : Inverted index dan TF-IDF")
    main.calculate_tfidf_for_words(unique_words_all, linked_list_data, files_with_index)
    dict_words, total_files, total_vocab, vecD = main.vec_d_matrix(unique_words_all, files_with_index, linked_list_data)
    #TODO :  Capture salah satu isi posting list dari sebuah kata pada dictionary.

    print("No 4 : Menghitung cosine similarity")
    vecQ = main.query_input(total_vocab, dict_words, linked_list_data)
    d_cosines = main.output_files(vecD, vecQ, files_with_index)

    print("No 5 : Nilai vektor similarity, dan nilai recall dan precision ")
    predicted_relevance, true_relevance = main.vector_relevance(d_cosines, files_with_index)
    recall, precision = main.hitung_recall_precision(predicted_relevance, true_relevance)


    print(" No 6 : Rochio Algorithm")
    alpha = 1
    beta = 0.75
    gamma = 0.25

    print("Menghitung relevance feedback")
    updated_query = main.hitung_rochio2(vecQ,d_cosines, files_with_index, alpha, beta, gamma)

    # Calculate cosine similarity with the updated query
    d_cosines2 = main.output(vecD, updated_query)
    #TODO : capture hasil dokumen belum

    print("Nilai similarity setelah Rocchio")
    predicted_relevance, true_relevance = main.vector_relevance(d_cosines2, files_with_index)
    recall2, precision2 = main.hitung_recall_precision(predicted_relevance, true_relevance)


    print("\nComparison of Precision and Recall")
    print("Initial Query Precision:", precision)
    print("Initial Query Recall:", recall)

    print('Precision after Rocchio:', precision2)
    print('Recall after Rocchio:', recall2)




    