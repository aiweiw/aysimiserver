# coding=utf-8

import fasttext
import os
import re

"""
Use seg-file generate word-vectors
input: seg-file, 0: cbow - 1: skipgram
output: file-matrix every word by cbow or skipgram
"""


def learn_model(dir_model, file_built, model_method=0):
    """
    :param dir_model:
    :param file_built:
    :param model_method:
    :return:
    """
    if not os.path.exists(file_built):
        return None

    if not os.path.exists(dir_model):
        os.mkdir(dir_model)

    if dir_model[len(dir_model) - 1] != '/':
        dir_model += '/'

    file_regex_match = re.search(r'\d+', file_built)
    date_span = file_regex_match.span()

    if model_method == 0:
        model_name = 'cbowAnsjModel' + file_built[date_span[0]:date_span[1]]
        fasttext.cbow(file_built, dir_model + model_name)
    else:
        model_name = 'skipGramAnsjModel' + file_built[date_span[0]:date_span[1]]
        fasttext.skipgram(file_built, dir_model + model_name)


def load_model(file_model):
    """
    :param file_model:
    :return:
    """
    if not os.path.exists(file_model):
        return None

    return fasttext.load_model(file_model)
