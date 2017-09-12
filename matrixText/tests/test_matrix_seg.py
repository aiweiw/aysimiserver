# -*- coding: utf-8 -*-

import re
import time
import sys

sys.path.append('..')

from matrix_seg import seg_file, seg_xml_ansj, list_file, seg_sentence, patt_time, \
    seg_cont_ansj, seg_xml_ansj_dm

if __name__ == "__main__":

    if sys.getdefaultencoding() != 'utf-8':
        reload(sys)
        sys.setdefaultencoding('utf8')

    # seg_path = '../../matTextData/dataTemp/dataTfidfTest'
    # files = list_file(seg_path)
    # for file_name in files:
    #     seg_file(seg_path, file_name, '../../matTextData/dataTemp/segFileTest',
    #              'ansj', 'http://127.0.0.1:8080/SegService/servlet/Segment')

    # print seg_cont_ansj('北京市高级人民法院', 'http://172.23.4.14:8080/SegService/servlet/Segment')
    # print seg_cont_ansj('北京市高级人民法院', 'http://127.0.0.1:8080/SegService/servlet/Segment')


    # ansj-seg-content
    # seg_xml_ansj('../../anyou/AY_xingshi.xml', '../../anyou/xingshi/src/', '../../matTextData/dataTemp/ansjSeg/',
    #              'http://172.23.4.9:8080/SegService/servlet/Segment')
    startTime = time.time()
    seg_xml_ansj_dm('../../anyou/AY_xingshi.xml', '../../w2vdata/xingshi/src/ssjlPart/',
                    '../../w2vdata/cache/xingshi/ssjlTestJa/', 'http://172.23.4.32:8080/SegService/servlet/Segment')

    print '------: ', time.time() - startTime


    print 'ok'
