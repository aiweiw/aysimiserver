# coding=utf-8

import jieba.posseg as pseg
import node
import os
import time
import re

minshi_firstlist = []
minshi_nodemap = {}
minshi_label_map = {}

node.loadConfig('../anyou/AY_minshi.xml', minshi_firstlist, minshi_nodemap, minshi_label_map)

xingshi_firstlist = []
xingshi_nodemap = {}
xingshi_label_map = {}

node.loadConfig('../anyou/AY_xingshi.xml', xingshi_firstlist, xingshi_nodemap, xingshi_label_map)

# pre-data
anyou_dict_txt = {}
for anyou in xingshi_firstlist:
    #
    anyou_list_txt = []
    node.get_all_grandsons_id(anyou, anyou_list_txt)
    anyou_key = anyou.get('DM')
    anyou_dict_txt[anyou_key] = list(set(anyou_list_txt))

anyou_dict_minshi = {}
for anyou in minshi_firstlist:
    anyou_list_txt = []
    node.get_all_grandsons_id(anyou, anyou_list_txt)
    anyou_key = anyou.get('DM')
    anyou_dict_minshi[anyou_key] = list(set(anyou_list_txt))

print len(anyou_dict_txt)


# generate anyou data-to-learn word-vector
# mingshi_data_dir = '../anyou/xingshi/src/'

# for key, val in anyou_dict_txt.items():
#     #
#     train_data_dir = 'dataTemp/'
#     if not os.path.exists(train_data_dir):
#         os.mkdir(train_data_dir)
#
#     train_data_file = train_data_dir + key + '.txt.learn'
#
#     try:
#         train_file = open(train_data_file, 'w')
#         for v in val:
#             #
#             tmp_file = mingshi_data_dir + str(v) + '.txt'
#             if os.path.exists(tmp_file):
#                 tmp_file_read = open(tmp_file)
#                 line = tmp_file_read.readline()
#                 while line:
#                     #
#                     # words = jieba.cut(line, cut_all=False)
#                     words = pseg.cut(line)
#
#                     for word, flag in words:
#                         if word == '\n':
#                             train_file.write(word.encode('utf-8'))
#                         else:
#                             train_file.write((word + ' ').encode('utf-8'))
#                         train_file.flush()
#                     line = tmp_file_read.readline()
#                 tmp_file_read.close()
#                 time.sleep(0)
#                 print 'on...'
#
#         train_file.close()
#         time.sleep(0)
#         print '---going---'
#
#     except Exception, e:
#         print Exception, e
#     finally:
#         if train_file and not train_file.closed:
#             train_file.close()
#         if tmp_file_read and not tmp_file_read.closed:
#             tmp_file_read.close()

re_file = re.compile(r'^9\d{3}')
fm_data_dir = 'dataTemp/'
fm_data_file = fm_data_dir + 'fm.txt'
fm_data_obj = open(fm_data_file, 'w')
i = 0
for key, val in minshi_nodemap.items():
    if val.get('YZ') == '0' and re_file.match(val.get('DM')):
        fm_data_obj.write(val.get('DM') + '\r\n')
        i += 1
        print '------ ', i

fm_data_obj.close()

print 'ok'
