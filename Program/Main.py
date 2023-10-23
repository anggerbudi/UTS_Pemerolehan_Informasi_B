"""
Kelas       : Pemerolehan Informasi B
Tugas       : Program Tugas UTS
Kelompok    : Martinus Angger Budi Wicaksono    (215314163)
              Antonius Yoga Chris Raharja       (215314175)
              Antonius Miquel Aureliano         (215314181)
"""


from VectorSpaceModel import Program

if __name__ == "__main__":
    # file_folder = "C:/Users/angger/PycharmProjects/PemerolehanInformasi/TextFileTest/*"
    file_folder = "C:/Users/angger/PycharmProjects/PemerolehanInformasi/TextFile/*"
    main = Program(file_folder)
    unique_words_all, files_with_index = main.text_loader_and_words_collector()
    linked_list_data = main.process_data_and_update_linked_list(unique_words_all)
    main.calculate_tfidf_for_words(unique_words_all, linked_list_data, files_with_index)
    dict_words, total_files, total_vocab, vecD = main.vec_d_matrix(unique_words_all, files_with_index, linked_list_data)
    print(files_with_index)

    vecQ = main.query_input(total_vocab, dict_words, linked_list_data)
    main.output(vecD, vecQ, total_files, files_with_index)
