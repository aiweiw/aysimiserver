#!/usr/bin/env python
# -*- coding:utf-8 -*-
import codecs
import os
import sys
import time
import urllib
import urllib2
from multiprocessing import Pool

sys.path.append('..')
import node

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf8')

max_link_time = 20

# file_xml_path = os.path.expanduser('~/fastText/anyou/AY_xingshi.xml')
# to_seg_path = os.path.expanduser('~/fastText/w2vdata/xingshi/src/ssjlPart/')
# save_seg_file = os.path.expanduser('~/fastText/w2vdata/cache/xingshi/ssjl/')
# ansj_serve_url = 'http://127.0.0.1:8080/SegService/servlet/Segment'

file_xml_path = None
to_seg_path = None
save_seg_file = None
ansj_serve_url = None

def seg_cont_ansj(to_seg_content, ansj_serve_url=None, out_filter=0):
    """
    :param to_seg_content:
    :param ansj_serve_url:
    :param out_filter: 0 - seg_vec, 1 - seg_filter, * - seg_vec, seg_filter
    :return:
    """
    if not to_seg_content or not ansj_serve_url:
        return None

    seg_ontent = dict()
    seg_ontent['segTent'] = to_seg_content
    data_urlencode = urllib.urlencode(seg_ontent)
    req = urllib2.Request(ansj_serve_url, data=data_urlencode)

    i_link = 0
    response = None
    seg_res = ''
    while i_link < max_link_time:
        try:
            response = urllib2.urlopen(req)
            if response:
                seg_res = response.read()
                if '-SEGMENT-' in seg_res:
                    if i_link > 0:
                        print 'link time: ', i_link
                    #     print 'src:', to_seg_content
                    #     print 'link time: ', i_link, ' : ', seg_res
                    i_link = 0
                    break
                response = None
                seg_res = ''
            i_link += 1
        except urllib2.URLError, e:
            i_link += 1
            if hasattr(e, 'reason'):
                print 'Failed to reach the server'
                print 'The reason:', e.reason
            elif hasattr(e, 'code'):
                print "The server couldn't fulfill the request"
                print 'Error code:', e.code
                print 'Return content:', e.read()

    # seg_res = response.read()
    seg_vec_tfidf = seg_res.split('-SEGMENT-')

    if out_filter == 0:
        # return segseg-words full
        return seg_vec_tfidf[0]
    elif out_filter == 1:
        # return seg-words filter
        return seg_vec_tfidf[1]
    else:
        # return seg-words full+filter
        return seg_vec_tfidf[0], seg_vec_tfidf[1]


def xml_anyou_dm(xml_file):
    """
    :param xml_file:
    :return:
    """
    if not os.path.exists(xml_file):
        return list()

    anyou_firstlist = []
    anyou_nodemap = {}
    anyou_label_map = {}

    node.loadConfig(xml_file, anyou_firstlist, anyou_nodemap, anyou_label_map)

    anyou_dict_txt = {}
    for key in anyou_nodemap.keys():
        if key is None:
            continue
        anyou_list_txt = []
        node.get_all_grandsons_id(anyou_nodemap.get(key), anyou_list_txt)
        anyou_dict_txt[key] = list(set(anyou_list_txt))

    list_anyou_dm = [[(k, v)] for k, v in anyou_dict_txt.items()]

    return list_anyou_dm


def seg_xml_ansj_dm(anyou_list_txt):
    """
    :param anyou_list_txt:
    :return:
    """
    global to_seg_path
    global save_seg_file
    global ansj_serve_url

    if not anyou_list_txt or not os.path.exists(to_seg_path) or not ansj_serve_url:
        return

    anyou_dict_txt = dict()
    for param in anyou_list_txt:
        anyou_dict_txt[param[0]] = param[1]

    for key, val in anyou_dict_txt.items():

        if not os.path.exists(save_seg_file):
            os.mkdir(save_seg_file)

        if save_seg_file[len(save_seg_file) - 1] != '/':
            save_seg_file += '/'
        train_data_file = save_seg_file + key + '.txt.ansj.learn'

        if_create_train_file = False
        for v in val:
            if_file = to_seg_path + str(v) + '.txt'
            if os.path.exists(if_file):
                if_create_train_file = True
                break
        if not if_create_train_file:
            continue

        try:
            train_file = codecs.open(train_data_file, 'a', 'utf-8')
            for v in val:
                tmp_file = to_seg_path + str(v) + '.txt'
                if os.path.exists(tmp_file):
                    print '---------- (key, val): (', key, v, ')'
                    tmp_file_read = open(tmp_file)
                    line = tmp_file_read.readline()

                    while line:

                        seg_vec_jion = seg_cont_ansj(line, ansj_serve_url)
                        seg_vec = seg_vec_jion.strip(' \r\n').split()

                        for word_to_vec in seg_vec:
                            word = ''.join(word_to_vec.split())

                            train_file.write((word + ' ').encode('utf-8'))
                            train_file.flush()

                        train_file.write(('\r\n').encode('utf-8'))
                        train_file.flush()

                        line = tmp_file_read.readline()

                    tmp_file_read.close()
                    time.sleep(0)
                    print 'on...'

            train_file.close()
            time.sleep(0)
            print '---going---'

        except Exception, e:
            print Exception, e
        finally:
            if train_file and not train_file.closed:
                train_file.close()
            if tmp_file_read and not tmp_file_read.closed:
                tmp_file_read.close()


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print 'argv error'
    else:
        print '---start---'
        for k in range(len(sys.argv)):
            print k, ' : ', sys.argv[k]
        start_time = time.time()
        file_xml_path = sys.argv[2]  # xml path
        to_seg_path = sys.argv[3]  # src dir
        save_seg_file = sys.argv[4]  # save segment data dir
        ansj_serve_url = sys.argv[5]  # word wegment server
        dict_xml_dm = xml_anyou_dm(file_xml_path)
        pool = Pool(int(sys.argv[1]))
        resultList = pool.map(seg_xml_ansj_dm, dict_xml_dm)
        pool.close()
        pool.join()

        print "used time is ", time.time() - start_time
