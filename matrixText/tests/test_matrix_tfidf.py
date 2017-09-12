# coding=utf-8

from matrixText.matrix_tfidf import tfidf_files, tfidf_file

if __name__ == "__main__":

    tfidf_files('../data/segAnsj/ansjSegXingshi', '../dataTemp/tfidfFileAnsj')
    # tfidf_file('../dataTemp/segFileTest/907901.txt.seg', '../dataTemp/tfidfFileTest')
    # tfidf_file('../data/9000.txt.learn', '../dataTemp/tfidfFileTest')
    print 'ok'
