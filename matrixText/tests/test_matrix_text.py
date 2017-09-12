# coding=utf-8

import os
import re
import time

from matrixText.matrix_text import matrix_anyou_vec_weight, matrix_similar_src, matrix_file_vec
from matrixText.matrix_learn import load_model

if __name__ == "__main__":
    file_obj = None
    file_model = '../data/wordVecAnsjSeg/dataModelXingshi/cbowAnsjModel1.bin'
    model = load_model(file_model)
    print 'load model'

    # generate anyou-9000...vector
    path = '../../anyou/xingshi/src/'
    # path = '../dataTemp/dataTfidfTest/'
    list_file = os.listdir(path)
    # re_file = re.compile(r'^9\d{3}')
    re_file = re.compile(r'^\d+')
    for f in list_file:
        if re_file.match(f):
            print 'match:', f
            file_obj = path + f
            dir_text_tfidf = dict()
            dir_text_tfidf['dir_text'] = '../dataTemp/cbowVectAnsj/'
            dir_text_tfidf['text_tfidf'] = '../dataTemp/tfidfTextAnsj/'
            dir_text_tfidf['dir_tfidf'] = '../data/tfidfAnsjSeg/tfidfFileAnsjXs/'
            matrix_anyou_vec_weight(file_obj=file_obj, dir_text_tfidf=dir_text_tfidf, word_vec_model=model, words_max_tfidf=1,
                                    ansj_serve_url='http://172.23.4.9:8080/SegService/servlet/Segment', sen_vec=0)
        else:
            print 'no:', f

    # src-vec-sample
    # file_src = '../dataTemp/similarTest.txt'
    # obj_src = open(file_src, 'r')
    # src_line = obj_src.readline()
    # obj_src.close()
    # time.sleep(0)
    # file_vec = '../dataTemp/vectemp.txt'
    # matrix_similar_src(content=src_line, file_vec=file_vec, file_model=file_model,
    #                    ansj_serve_url='http://172.23.4.9:8080/SegService/servlet/Segment', dim=100)
    # vec_text = matrix_file_vec(file_obj=file_obj, file_model=file_model,
    #                            ansj_serve_url='http://172.23.4.9:8080/SegService/servlet/Segment', dim=100)

    # test-vec-sample
    # file_obj = '../data/similarTest.txt'
    # file_text_tfidf = dict()
    # file_text_tfidf['file_text'] = '../dataTemp/vectemp.txt'
    # file_text_tfidf['file_tfidf'] = ''
    # matrix_text_file(file_obj=file_obj, file_text_tfidf=file_text_tfidf, file_model=file_model, sen_vec=0, dim=100)
    # vec_text = matrix_text_vec(file_obj=file_obj, file_text_tfidf=file_text_tfidf, file_model=file_model)

    # anyou-vec-model
    # file_obj = '../../anyou/minshi/src/9265.txt'
    # dir_text_tfidf = dict()
    # dir_text_tfidf['dir_text'] = '../dataTemp/'
    # dir_text_tfidf['text_tfidf'] = '../dataTemp/textTfidf/'
    # dir_text_tfidf['dir_tfidf'] = '../tfidfFile/'
    # matrix_text_file(file_obj=file_obj, file_text_tfidf=dir_text_tfidf, file_model=file_model, words_max_tfidf=1,
    #                  ansj_serve_url='http://172.23.4.9:8080/SegService/servlet/Segment', sen_vec=0)

    # matrix_text(file_obj=file_obj, file_model=file_model, sen_vec=1)

    print 'ok'
