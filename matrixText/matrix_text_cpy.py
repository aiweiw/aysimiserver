# coding=utf-8

import os
import re
import numpy as np
import time
import sys

from matrix_learn import load_model
from matrix_seg import seg_content, seg_sentence


def matrix_line(content, model, tfidf=None, norm_tfidf=None, tfidf_max_num=1, sen_vec_method=0, dim=100):
    """
    :param content:
    :param model:
    :param tfidf:
    :param norm_tfidf:
    :param tfidf_max_num:
    :param sen_vec_method:
    :param dim:
    :return:
    """
    if not content or not model:
        return None

    if sys.getdefaultencoding() != 'utf-8':
        reload(sys)
        sys.setdefaultencoding('utf-8')

    line_list = seg_sentence(content)

    matrix_content = list()
    content_tfidf = list()

    for line_one in line_list:
        if line_one == '\n':
            continue
        line_vec = np.zeros(dim)
        line_tfidf = list()
        word_num = 0
        words = seg_content(line_one)
        # TODO: update word-vector to sentence-vector by NN or ...
        for word, flag in words:
            if word == ' ':
                continue
            if sen_vec_method == 0:
                # mean-words to sentence-vector
                line_vec += np.array(model[word])
                word_num += 1
            elif sen_vec_method == 1 and tfidf and norm_tfidf:
                # zero tfidf dealwith no-words
                if word.encode('utf-8') in tfidf:
                    line_vec += np.array(model[word]) * tfidf[word.encode('utf-8')]
                    # else:
                    #     pass
            elif sen_vec_method == 2 and tfidf and norm_tfidf:
                # mininum tfidf dealwith no-words
                if word.encode('utf-8') in tfidf:
                    line_vec += np.array(model[word]) * tfidf[word.encode('utf-8')]
                else:
                    line_vec += np.array(model[word]) * norm_tfidf['mininum']
            elif sen_vec_method == 3 and tfidf and norm_tfidf:
                # normlize tfidf
                if word.encode('utf-8') in tfidf:
                    line_vec += np.array(model[word]) * (tfidf[word.encode('utf-8')] - norm_tfidf['mininum']) \
                                / (norm_tfidf['maxinum'] - norm_tfidf['mininum'])
                else:
                    line_vec += np.array(model[word]) * (norm_tfidf['average'] - norm_tfidf['mininum']) \
                                / (norm_tfidf['maxinum'] - norm_tfidf['mininum'])
            else:
                # default: mean-words-vector to sentence-vector
                line_vec += np.array(model[word])
                word_num += 1
                # pass

            if tfidf and word.encode('utf-8') in tfidf:
                line_tfidf.append(tfidf[word.encode('utf-8')])
            elif norm_tfidf:
                line_tfidf.append(norm_tfidf['mininum'])
            else:
                line_tfidf.append(0.0)

        if line_tfidf:
            content_tfidf.append(max(line_tfidf) / tfidf_max_num)
        else:
            content_tfidf.append(0.0)

        if not (sen_vec_method in [1, 2, 3] and tfidf and norm_tfidf):
            if word_num > 0:
                line_vec = line_vec / word_num
                # print word_num, line_vec
                # print '---do.mean.words-----'

        matrix_content.append(line_vec)

    return matrix_content, content_tfidf


def weight_tfidf(whole_tfidf_file, norm_tfidf_file):
    """
    :param whole_tfidf_file:
    :param norm_tfidf_file:
    :return:
    """
    if not (whole_tfidf_file and os.path.exists(whole_tfidf_file)) \
            or not (norm_tfidf_file and os.path.exists(norm_tfidf_file)):
        return None, None

    words_tfidf = dict()
    norm_tfidf = dict()
    file_tfidf = None
    file_avg_tfidf = None
    try:
        file_tfidf = open(whole_tfidf_file, 'r')
        line_tfidf = file_tfidf.readline()
        while line_tfidf:
            key_val = line_tfidf.strip(" \n").split(" ")
            words_tfidf[key_val[0]] = float(key_val[1])
            line_tfidf = file_tfidf.readline()
        file_tfidf.close()
        time.sleep(0)

        file_avg_tfidf = open(norm_tfidf_file, 'r')
        avg_line_tfidf = file_avg_tfidf.read()
        line_list = avg_line_tfidf.strip(" \n").split("\n")
        for line in line_list:
            data_list = line.split()
            if data_list[0] == 'average:':
                norm_tfidf['average'] = float(data_list[1])
            elif data_list[0] == 'maximum:':
                norm_tfidf['maxinum'] = float(data_list[1])
            elif data_list[0] == 'mininum:':
                norm_tfidf['mininum'] = float(data_list[1])
        file_avg_tfidf.close()
        time.sleep(0)
    except Exception, e:
        print Exception, e
    finally:
        if file_tfidf and not file_tfidf.closed:
            file_tfidf.close()
        if file_avg_tfidf and not file_avg_tfidf.closed:
            file_avg_tfidf.close()

    return words_tfidf, norm_tfidf


def matrix_text_dir(file_obj=None, dir_text_tfidf=None, file_model=None, words_max_tfidf=1, sen_vec=0, dim=100):
    """
    :param file_obj:
    :param dir_text_tfidf:
    :param file_model:
    :param words_max_tfidf:
    :param sen_vec:
    :param dim:
    :return:
    """
    if not (file_obj and os.path.exists(file_obj)) or not (file_model and os.path.exists(file_model)):
        return

    if not (dir_text_tfidf and len(dir_text_tfidf) == 3):
        return

    file_object = None
    file_vector = None
    content_tfidf = None
    try:
        file_ti = None
        file_avg_ti = None
        file_vec = None
        content_ti = None
        if dir_text_tfidf:
            # make anyou-vec-text
            date_span = re.search(r'\d+', file_obj).span()
            if not os.path.exists(dir_text_tfidf['dir_text']):
                os.mkdir(dir_text_tfidf['dir_text'])
            file_vec = dir_text_tfidf['dir_text'] + file_obj[date_span[0]:date_span[1]] + '.txt.out.vect'
            if not os.path.exists(dir_text_tfidf['text_tfidf']):
                os.mkdir(dir_text_tfidf['text_tfidf'])
            content_ti = dir_text_tfidf['text_tfidf'] + file_obj[date_span[0]:date_span[1]] + '.txt.content.tfidf'
            if dir_text_tfidf['dir_tfidf']:
                file_ti = dir_text_tfidf['dir_tfidf'] + file_obj[date_span[0]:date_span[1]] + '.txt.tfidf'
                file_avg_ti = dir_text_tfidf['dir_tfidf'] + 'average.txt.tfidf'

        words_tfidf, norm_tfidf = weight_tfidf(file_ti, file_avg_ti)

        file_object = open(file_obj)
        file_vector = open(file_vec, 'w')
        content_tfidf = open(content_ti, 'w')
        line = file_object.readline()
        # model-load-outside
        model = load_model(file_model)
        # print 'load model'

        while line and model:

            matrix_content, vec_tfidf = matrix_line(content=line, model=model, tfidf=words_tfidf,
                                                    norm_tfidf=norm_tfidf, tfidf_max_num=words_max_tfidf,
                                                    sen_vec_method=sen_vec, dim=dim)

            if matrix_content and vec_tfidf:

                # model-mean-anyou
                # sum_line = np.zeros(dim)
                # for line in matrix_content:
                #     sum_line += line
                # sum_line = sum_line / len(matrix_content)
                # for one_num in sum_line:
                #     file_vector.write(str(one_num) + ' ')
                # file_vector.write('\n')

                # model-every-line
                for line_vec in matrix_content:
                    for number in line_vec:
                        file_vector.write(str(number) + ' ')
                file_vector.write('\n')

                for line_vec in vec_tfidf:
                    content_tfidf.write(str(line_vec) + ' ')
                content_tfidf.write('\n')

            line = file_object.readline()
            # print '---on---'

        file_object.close()
        time.sleep(0)
        file_vector.close()
        time.sleep(0)
        content_tfidf.close()
        time.sleep(0)

    except Exception, e:
        print Exception, e
    finally:
        if file_object and not file_object.closed:
            file_object.close()
        if file_vector and not file_vector.closed:
            file_vector.close()
        if content_tfidf and content_tfidf.closed:
            content_tfidf.close()


def matrix_text_file(file_obj=None, file_text_tfidf=None, file_model=None, sen_vec=0, dim=100):
    """
    :param file_obj:
    :param file_text_tfidf:
    :param file_model:
    :param sen_vec:
    :param dim:
    :return:
    """
    if not (file_obj and os.path.exists(file_obj)) or not (file_model and os.path.exists(file_model)):
        return

    if not (file_text_tfidf and len(file_text_tfidf) == 2):
        return

    file_object = None
    file_vector = None
    try:
        file_ti = None
        file_avg_ti = None
        file_vec = None
        if file_text_tfidf:
            # make test-vec-sample
            file_vec = file_text_tfidf['file_text']
            if file_text_tfidf['file_tfidf']:
                file_ti = file_text_tfidf['file_tfidf']
                file_avg_ti = file_ti[:(file_ti.rindex('/') + 1)] + 'average.txt.tfidf'

        words_tfidf, norm_tfidf = weight_tfidf(file_ti, file_avg_ti)

        file_object = open(file_obj)
        file_vector = open(file_vec, 'w')
        line = file_object.readline()
        model = load_model(file_model)
        # print 'load model'

        while line and model:

            matrix_content, vec_tfidf = matrix_line(content=line, model=model, tfidf=words_tfidf,
                                                    norm_tfidf=norm_tfidf, sen_vec=sen_vec, dim=dim)

            if matrix_content:

                # model-mean-anyou
                # sum_line = np.zeros(dim)
                # for line in matrix_content:
                #     sum_line += line
                # sum_line = sum_line / len(matrix_content)
                # for one_num in sum_line:
                #     file_vector.write(str(one_num) + ' ')
                # file_vector.write('\n')

                # model-every-line
                for line_vec in matrix_content:
                    for number in line_vec:
                        file_vector.write(str(number) + ' ')
                file_vector.write('\n')

            line = file_object.readline()
            # print '---on---'

        file_object.close()
        time.sleep(0)
        file_vector.close()
        time.sleep(0)

    except Exception, e:
        print Exception, e
    finally:
        if file_object and not file_object.closed:
            file_object.close()
        if file_vector and not file_vector.closed:
            file_vector.close()


def matrix_src(content, file_vec=None, file_model=None, dim=100):
    """
    :param content:
    :param file_vec:
    :param file_model:
    :param dim:
    :return:
    """
    if not content or not file_vec or not (file_model and os.path.exists(file_model)):
        return


    file_vector = None
    try:
        file_vector = open(file_vec, 'w')
        model = load_model(file_model)
        # print 'load model'

        matrix_content, vec_tfidf = matrix_line(content=content, model=model, dim=dim)

        if matrix_content:

            # model-every-line
            for line_vec in matrix_content:
                for number in line_vec:
                    file_vector.write(str(number) + ' ')
            file_vector.write('\n')

        file_vector.close()
        time.sleep(0)

    except Exception, e:
        print Exception, e
    finally:
        if file_vector and not file_vector.closed:
            file_vector.close()


def matrix_text_vec(file_obj=None, file_text_tfidf=None, file_model=None, dim=100):
    """
    :param file_obj:
    :param file_text_tfidf:
    :param file_model:
    :param dim:
    :return:
    """
    if not (file_obj and os.path.exists(file_obj)) or not (file_model and os.path.exists(file_model)):
        return

    if not (file_text_tfidf and len(file_text_tfidf) == 2):
        return

    file_object = None
    matrix_content = []
    try:
        file_ti = None
        file_avg_ti = None
        if file_text_tfidf and file_text_tfidf['file_tfidf']:
            file_ti = file_text_tfidf['file_tfidf']
            file_avg_ti = file_ti[:(file_ti.rindex('/') + 1)] + 'average.txt.tfidf'
        words_tfidf, norm_tfidf = weight_tfidf(file_ti, file_avg_ti)

        file_object = open(file_obj)
        line = file_object.readline()
        model = load_model(file_model)
        # print 'load model'

        if line and model:
            matrix_content, vec_tfidf = matrix_line(content=line, model=model,
                                                    tfidf=words_tfidf, norm_tfidf=norm_tfidf, dim=dim)

        file_object.close()
        time.sleep(0)

    except Exception, e:
        print Exception, e
    finally:
        if file_object and not file_object.closed:
            file_object.close()

    return matrix_content
