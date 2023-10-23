"""
Kelas       : Pemerolehan Informasi B
Tugas       : Program Tugas UTS
Kelompok    : Martinus Angger Budi Wicaksono    (215314163)
              Antonius Yoga Chris Raharja       (215314175)
              Antonius Miquel Aureliano         (215314181)
"""


class NodeClass:
    def __init__(self, doc_id, freq=0):
        self.freq = freq
        self.tfidf = 0
        self.doc = doc_id
        self.next_node = None


class LinkedListClass:
    def __init__(self, head=None, num_docs=0):
        self.head = head
        self.num_docs = num_docs
        self.idf = 0
