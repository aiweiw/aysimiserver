# coding=utf-8

import os
import sys
import re
import time
import gc

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

"""
Compute tfidf by sklearn
input: all seg-files to compute
output: tfidf - every file every word
"""


# the dir-all-files to tfidf, then save every one
def tfidf_files(seg_path, tfidf_dir):
    """
    :param seg_path:
    :param tfidf_dir:
    :return:
    """

    if not os.path.exists(seg_path):
        return

    if seg_path[len(seg_path) - 1] != '/':
        seg_path += '/'

    if sys.getdefaultencoding() != 'utf-8':
        reload(sys)
        sys.setdefaultencoding('utf8')

    try:
        list_file = os.listdir(seg_path)
        corpus = []
        # for ff in list_file:
        for i in range(len(list_file)):
            fname = seg_path + list_file[i]
            f = open(fname, 'r')
            content = f.read()
            f.close()
            time.sleep(0)
            corpus.append(content)

        # make matrix-tfidf
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        time.sleep(0)
        del corpus
        word = vectorizer.get_feature_names()
        # weight = tfidf.toarray()  # matrix_tfidf[len(list_file), len(words)]-need big memory
        weight_coo = tfidf.tocoo()
        time.sleep(0)
        del tfidf
        gc.collect()
        time.sleep(0.001)
        print 'gc.collect'

        # file-number
        doc_word_num = {-1: -1}
        if not list_file:
            return
        elif len(list_file) == 1:
            doc_word_num[0] = len(weight_coo.row) - 1
        else:
            for i in range(0, len(list_file) - 1):
                if i == 0:
                    doc_word_num[i] = list(weight_coo.row).index(i + 1) - 1
                else:
                    # doc_word_num[i] = list(weight_coo.row).index(i + 1) - 1
                    doc_word_num[i] = list(weight_coo.row).index(i + 1, doc_word_num[i - 1]) - 1
                    # print '---store---: ', i, doc_word_num[i]
                    # print '---check---: ', i, list(weight_coo.row).index(i + 1) - 1
            doc_word_num[i + 1] = len(weight_coo.row) - 1
            # print '---store---: ', i + 1, doc_word_num[i + 1]
            # print '---check---: ', i, len(weight_coo.row) - 1


        if not os.path.exists(tfidf_dir):
            os.mkdir(tfidf_dir)
        # i = 0  # file serial
        # for ff in list_file:
        for i in range(len(list_file)):
            date_span = re.search(r'\d+', list_file[i]).span()
            file_tfidf = tfidf_dir + '/' + list_file[i][date_span[0]:date_span[1]] + '.txt.ansj.tfidf'
            print u'--------Writing all the tf-idf in the', i, u' file into ', file_tfidf, '--------'

            f = open(file_tfidf, 'w')
            # method-i
            # for j in range(len(word)):
            #     # method-0
            #     # list_row = [key for key, val in enumerate(list(weight_coo.row)) if val == i]
            #     # list_col = [key for key, val in enumerate(list(weight_coo.col)) if val == j]
            #     # mark = list(set([k for k, v in enumerate(list(weight_coo.row)) if v == i]).intersection(
            #     #     set([k for k, v in enumerate(list(weight_coo.col)) if v == j])))
            #
            #     # method-1
            #     mark = None
            #     for k, v in enumerate(list(weight_coo.col)):
            #         if v == j and doc_word_num[i-1] < k <= doc_word_num[i]:
            #             mark = k
            #             break
            #
            #     if mark:
            #         f.write(word[j].encode('utf-8') + ' '.encode('utf-8') +
            #                 str(weight_coo.data[mark]).encode('utf-8') + '\r\n')
            #     else:
            #         f.write(word[j].encode('utf-8') + ' '.encode('utf-8') +
            #                 str(0.0).encode('utf-8') + '\r\n')

            # method-ii
            # norm = weight_coo.data[doc_word_num[i - 1] + 1: doc_word_num[i] + 1]
            # norm_sum_sqr = sum(map(lambda a: a ** 2, norm))
            for j in range(doc_word_num[i - 1] + 1, doc_word_num[i] + 1, 1):
                f.write((word[weight_coo.col[j]] + ' ' + str(weight_coo.data[j]) + '\r\n').encode('utf-8'))

            f.close()
            time.sleep(0)

        print u'--------word_num: ', len(weight_coo.data), u'avg_weight: ', sum(weight_coo.data) / len(weight_coo.data)
        file_tfidf = tfidf_dir + '/' + 'average.txt.ansj.tfidf'
        f = open(file_tfidf, 'w')
        f.write(('average: ' + str(sum(weight_coo.data) / len(weight_coo.data)) + '\r\n').encode('utf-8'))
        f.write(('maxinum: ' + str(max(weight_coo.data)) + '\r\n').encode('utf-8'))
        f.write(('mininum: ' + str(min(weight_coo.data)) + '\r\n').encode('utf-8'))
        f.close()
        time.sleep(0)
    except Exception, e:
        print Exception, e
    finally:
        pass

"""
Compute tfidf by sklearn
input: one-file to compute, one-line-one-kind (pre-generate)
output: tfidf - every word
"""

def tfidf_file(seg_file, tfidf_dir):
    """
    :param seg_file:
    :param tfidf_dir:
    :return:
    """

    if not os.path.exists(seg_file):
        return

    if sys.getdefaultencoding() != 'utf-8':
        reload(sys)
        sys.setdefaultencoding('utf8')

    try:
        corpus = []
        f = open(seg_file, 'r')
        line = f.readline()
        while line:
            corpus.append(line)
            line = f.readline()
        f.close()
        time.sleep(0)

        # make matrix-tfidf
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        del corpus
        word = vectorizer.get_feature_names()
        # weight = tfidf.toarray()  # matrix_tfidf[len(list_file), len(words)]-need big memory
        weight_coo = tfidf.tocoo()
        del tfidf
        gc.collect()
        time.sleep(0.001)
        print 'gc.collect'

        if not os.path.exists(tfidf_dir):
            os.mkdir(tfidf_dir)
        date_span = re.search(r'\d+', seg_file).span()
        file_tfidf = tfidf_dir + '/' + seg_file[date_span[0]:date_span[1]] + '.txt.tfidf'
        print u'--------Writing the tf-idf in the', seg_file, u' file into ', file_tfidf, '--------'

        # f = open(file_tfidf, 'w')
        # for i in range(len(weight_coo.row)):
        #     f.write((word[weight_coo.col[i]] + ' ' + str(weight_coo.data[i]) + '\r\n').encode('utf-8'))
        # f.close()
        time.sleep(0)
    except Exception, e:
        print Exception, e
    finally:
        pass