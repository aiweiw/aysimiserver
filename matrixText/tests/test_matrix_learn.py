# -*- coding: utf-8 -*-

from matrixText.matrix_learn import *

if __name__ == "__main__":

    # learn_model('../../seqnndata/dataModel', '../../seqnndata/duplicate.ml', 0)
    fasttext.cbow('../../seqnndata/duplicate.learn', '../../seqnndata/dataModel/' + 'duplicate')

    # model_load = load_model('../dataTemp/dataModel/cbowAnsjModel9000.bin')
    #
    # if model_load:
    #     print sum(model_load['原告']), model_load['原告']
    #     print sum(model_load['诉称']), model_load['诉称']

    print 'ok'
