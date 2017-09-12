# coding= utf-8
import codecs
import requests
import urllib2
from decimal import Decimal

import fasttext
import os
import re
from matrixText.matrix_multiply import *
import numpy as np
import time

import sys

reload(sys)
sys.setdefaultencoding('utf8')


# model_load = fasttext.load_model('data/model896.bin')
#
# print sum(model_load['原告']), model_load['原告']
# print sum(model_load['刘汉国']), model_load['刘汉国']
# print sum(model_load['冯文积']), model_load['冯文积']
# print sum(model_load['周道全']), model_load['周道全']
# print sum(model_load['张荣花']), model_load['张荣花']
# print sum(model_load['唐丽']), model_load['唐丽']
# print sum(model_load['卢福江']), model_load['卢福江']
# print sum(model_load['黄荷珠']), model_load['黄荷珠']
# print sum(model_load['诉称']), model_load['诉称']

# predictcmd = '/home/uww/fastText/fasttext print-vectors ' + \
#              'data/model896.bin < data/test.txt'
#
# print(predictcmd)
# cmdresult = os.popen(predictcmd).readlines();

# a = [[1., 2., 3.],
#      [12., 23., 34.],
#      [21., 42., 53.],
#      [41., 22., 63.],
#      [91., 32., 73.]]
#
# b = [[4., 5., 7.],
#      [41., 82., 73.],
#      [61., 62., 83.],
#      [81., 72., 93.],
#      [101., 32., 43.],
#      [41., 52., 73.],
#      [51., 22., 93.],
#      [71., 12., 23.]]
#
# c = [[(4., 5., 7.),
#      (41., 82., 73.),
#      (61., 62., 83.),
#      (81., 72., 93.)],
#      (101., 32., 43.),
#      [(41., 52., 73.),
#      (51., 22., 93.),
#      (71., 12., 23.)]]
#
# d, e = dist_matrix(a, b, 7)
# result = 0.0
#
# for i in range(len(d)):
#      result += corr_distance(d[i], e[i])
#
# result /= len(d)
# print result

# file_obj = 'data/vectemp_test.txt'
# file_vec = open(file_obj, 'r')
# line = file_vec.readline()
# while line:
#      list_number = line.rstrip(" \n").split(" ")
#      src_vec = []
#      for list_one in list_number:
#           src_vec.append(float(list_one))
#           time.sleep(0)
#      line = file_vec.readline()
#
# arr_list_number = np.array(list_number)
# print 'shape: ', arr_list_number.shape
# src_vec_two = np.array(src_vec).reshape(-1, 10)
# print src_vec_two
#
#
# del_num_src = [ num for num in src_vec if num > 0.00001 ]
# print del_num_src
# print len(del_num_src)
#
# [v for k, v in enumerate(list([11, 12, 13])) if v == 11 and v in [14, 15, 11]]

# test_dict = {'门户': '0.002', '金华': '0.003'}
# test_word = ['门户', '金华', '阿尔滨']
# for w in test_word:
#      if w in test_dict:
#           print float(test_dict[w])
#      else:
#           print 0.0

# x = [10, 2, 3.0]
# y = [4, 5, 6.0]
# c = np.array(x)*np.array(y)
#
# print type(c)
# print list(c)
# print type(list(c))
#
# d = map(lambda (a, b): a * b, zip(x, y))
#
# print d
# print type(d)

# def corr_distance(x, y):
#     """
#     :param x: test data
#     :param y: norm data
#     :return: corr-distance
#     """
#     if len(x) != len(y):
#         return
#
#     x_average = sum(x) / len(x)
#     y_average = sum(y) / len(x)
#
#     result1 = 0.0
#     result2 = 0.0
#     result3 = 0.0
#     for i in range(len(x)):
#         x_value = x[i] - x_average
#         y_value = y[i] - y_average
#         result1 += x_value * y_value
#         result2 += x_value ** 2
#         result3 += y_value ** 2
#
#     print '---1,2,3,4---:', result1, result2, result3, ((result2 * result3) ** 0.5)
#     result = result1 / ((result2 * result3) ** 0.5)
#     if math.isnan(result):
#         return 0.0
#
#     return result



# def corr_dist_py(x, y):
#      """
#      :param x:
#      :param y:
#      :return:
#      """
#      if len(x) != len(y):
#           return
#
#      x_average = sum(x) / len(x)
#      y_average = sum(y) / len(x)
#
#      average = (sum(x) + sum(y)) / (len(x) + len(y))
#
#      x_value = map(lambda x: x - average, x)
#      y_value = map(lambda y: y - average, y)
#
#      result1 = map(lambda (a, b): a * b, zip(x_value, y_value))
#      result2 = map(lambda a: a ** 2, x_value)
#      result3 = map(lambda b: b ** 2, y_value)
#
#      print '---1,2,3,4---:', sum(result1), sum(result2), sum(result3), ((sum(result2) * sum(result3)) ** 0.5)
#
#      return sum(result1) / (sum(result2) * sum(result3)) ** 0.5
#
#
# # print corr_distance(x, y)
# # print corr_dist_py(x, y)
#
# a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# b = [[10, 1, 2,], [11, 3, 4], [12, 5, 7]]
#
# flatten_a = flatten_list(a)
# flatten_b = flatten_list(b)
#
# sum_dist = 0.0
# for i in range(len(a)):
#      sum_dist += corr_dist_py(a[i], b[i])
# sum_dist /= len(a)
#
# print sum_dist
# print type(flatten_a)
#
# print corr_dist_py(list(flatten_a), list(flatten_b))

# a = (None, None)
# if a:
#      print 'a'
# else:
#      print 'no'

# a = '原告秦皇岛日飞昕虹仪器仪表有限公司诉称，原、被告于2009年5月5日、2009年6月12日分别签订了《技术服务合同》，' \
#     '约定由原告为被告所承包的黄骅港工程提供海上施工作业中的测量、定位服务。合同签订后，原告按约定从2009年5月至12月期间为被告提供了相应技术服务，' \
#     '合计应收费45．3万元。但被告实际给付费用17．2万元，并于2011年期间给付5万元，尚欠23．1万元至今未付，原告就欠款事宜多次找被告催要，' \
#     '被告一直推拖至今。故原告诉至法院，要求判令被告给付服务费人民币23．1万元及利息（自2010年1月1日起至判决生效日止，按中国人民银行同期贷款利率计算）'
#
# match = re.compile(r'\。|\，|\,|\.')
# list_a = match.split(a)

# a = [1, 3, 4]
# print a.index(3)

# print os.path.exists('abc/')
# print os.path.exists('def/')
#
# if not os.path.exists('abc/') or not os.path.exists('def/'):
#     print 'None'
# else:
#     print 'exist'

# a = [3., 9., 0., 3., 5.]
# a.sort(reverse=True)
# print sum(a[0:3]) / 3
# print max(a)
# b = []
# b.append(sum(a[0:3]) / 3)
# b.append(max(a))
# print b


# import jieba
# import jieba.posseg as pseg
# import os
# import sys
# from sklearn import feature_extraction
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
#
# if __name__ == "__main__":
#     corpus = [u"我 来到 北京 清华大学",  # 第一类文本切词后的结果，词之间以空格隔开
#               u"他 来到 了 网易 杭研 大厦",  # 第二类文本的切词结果
#               u"小明 硕士 毕业 与 中国 科学院",  # 第三类文本的切词结果
#               u"我 爱 北京 天安门"]  # 第四类文本的切词结果
#     vectorizer = CountVectorizer(analyzer='word')  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
#     transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
#     tfidf = transformer.fit_transform(
#         vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
#     word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
#     weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
#
#     count = 0
#     for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
#         print u"-------这里输出第", i, u"类文本的词语tf-idf权重------"
#         for j in range(len(word)):
#             print word[j], weight[i][j]
#             if weight[i][j] > 0.0000001:
#                 count += 1
#
#     print weight
#     sum_weight = 0
#     for row in weight:
#         print sum(row)
#         sum_weight += sum(row)
#
#     avg_weight = sum_weight / count
#     print avg_weight



# from matrixText.matrix_text import weight_tfidf
# from matrixText.matrix_text import seg_sentence, load_model
#
# from operator import itemgetter
# import jieba
# import jieba.posseg as pseg
# import fasttext
#
# target_tfidf = '../data/tfidfAnsjSeg/tfidfFileAnsj/9265.txt.ansj.tfidf'
# target_tfidf_avg = '../data/tfidfAnsjSeg/tfidfFileAnsj/average.txt.ansj.tfidf'
# words_tfidf, avg_tfidf = weight_tfidf(target_tfidf, target_tfidf_avg)
#
# print type(words_tfidf), len(words_tfidf)
# print type(avg_tfidf)
#
# # sorted(words_tfidf.items(), key=lambda item:item[1], reverse=True)
# b = sorted(words_tfidf.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
#
# print type(b), len(b)
# for ele in b:
#     print type(ele), ele[0], ele[1]
#
# for key, val in words_tfidf.items():
#     print key, val




# a = 'sadfagfdaghkdfjlhalfdaouifhdasofhduiosahfiduaos就是人们在生产生活当中所经历的典型的富有多种意义的事件陈述' \
#     '为人们更好的适应案例情景提供很多方便'
#
# b = '原告 诉称 ， 原 、 被告 于 2009 年 5 月 5 日 、 2009 年 6 月 12 日 分别 签订 了 《 技术 服务 合同 》 ， ' \
#     '约定 由 原告 为 被告 所 承包 的 黄骅港 工程 提供 海上 施工 作业 中 的 测量 、 定位 服务 。 合同 签订 后 ， ' \
#     '原告 按 约定 从 2009 年 5 月 至 12 月 期间 为 被告 提供 了 相应 技术 服务 ， 合计 应 收费 45 ． 3 万元 。' \
#     ' 但 被告 实际 给付 费用 17 ． 2 万元 ， 并于 2011 年 期间 给付 5 万元 ， 尚 欠 23 ． 1 万元 至今 未付 ， ' \
#     '原告 就 欠款 事宜 多次 找 被告 催要 ， 被告 一直 推拖 至今 。 故 原告 诉至 法院 ， ' \
#     '要求 判令 被告 给付 服务费 人民币 23 ． 1 万元 及 利息 （ 自 2010 年 1 月 1 日 起至 判决 生效日 止 ， ' \
#     '按 中国人民银行 同期 贷款 利率 计算 ） '
#
# # list_line = seg_sentence(a)
# # b = '../tfidfFile/average.txt.tfidf'
# # d = b.index('/')
# # e = b.rindex('/')
# #
# # print b[:d], b[:e]
#
# anyou_dict = '../data/anyou.dic'
#
# corpus = []
# f = open(anyou_dict)
# content = f.read()
# f.close()
# time.sleep(0)
# corpus.append(content)
#
# jieba.load_userdict(anyou_dict)
#
# seg_out = '../data/seg_similar.txt'
# tmp_seg_out = open('../data/seg_similar2.txt', 'w')
# words = pseg.cut(b)
# # model = load_model('../data/seg_similar.bin')
# for word, flag in words:
#     # print word, flag
#     tmp_seg_out.write((word + ' ').encode('utf-8'))
# tmp_seg_out.close()

# import sys
#
# reload(sys)
# sys.setdefaultencoding('utf-8')
#
# a = [{'DM': '9265', 'id': 6547, 'DATASYM': [(2, 2, 1.0, 0.57999677145299999), (4, 4, 1.0, 0.57999677145299999), (9, 9, 1.0, 0.57999677145299999), (1, 1, 1.0, 0.57540817092600005), (6, 6, 1.0, 0.57540817092600005), (10, 10, 1.0, 0.57540817092600005), (12, 12, 0.99680937318002216, 0.57357225818340918), (0, 0, 0.81212983093683422, 0.47103267994403458)], '\xe7\x9b\xb8\xe4\xbc\xbc\xe5\xba\xa6': 0.97611740051460705, '\xe6\xa1\x88\xe7\x94\xb1': u'\u670d\u52a1\u5408\u540c\u7ea0\u7eb7', 'prob': '0.738281'}, {'DM': '9265', 'id': 1014, 'DATASYM': [(2, 2, 1.0, 0.57999677145299999), (4, 4, 1.0, 0.57999677145299999), (9, 9, 1.0, 0.57999677145299999), (1, 1, 1.0, 0.57540817092600005), (6, 6, 1.0, 0.57540817092600005), (10, 10, 1.0, 0.57540817092600005), (12, 12, 0.99680937318002216, 0.57357225818340918), (0, 0, 0.81212983093683422, 0.47103267994403458)], '\xe7\x9b\xb8\xe4\xbc\xbc\xe5\xba\xa6': 0.97611740051460705, '\xe6\xa1\x88\xe7\x94\xb1': u'\u670d\u52a1\u5408\u540c\u7ea0\u7eb7', 'prob': '0.738281'}, {'DM': '9200', 'id': 11079, 'DATASYM': [(7, 5, 0.94012101503799683, 0.61006878047075552), (10, 11, 0.92499060478864825, 0.60025026691644545), (12, 14, 0.92156312752214753, 0.59802608849407413), (4, 4, 0.91993462101278234, 0.59696930860694764), (8, 6, 0.89186632248733533, 0.57875506557078227), (6, 13, 0.86986968387208075, 0.56448087929070856), (1, 2, 0.9282373566763118, 0.50886727772733464), (11, 12, 0.92662559410693657, 0.50798369636194174)], '\xe7\x9b\xb8\xe4\xbc\xbc\xe5\xba\xa6': 0.91540104068802997, '\xe6\xa1\x88\xe7\x94\xb1': u'\u627f\u63fd\u5408\u540c\u7ea0\u7eb7', 'prob': '0.126953'}, {'DM': '9200', 'id': 13075, 'DATASYM': [(12, 30, 0.97061378565691103, 0.62985632599637442), (4, 3, 0.95964305063025479, 0.62273713300789579), (8, 25, 0.9442371672644938, 0.61273985784152307), (1, 1, 0.93592115953398802, 0.60734338588378256), (7, 24, 0.88982912260649527, 0.57743307401118604), (6, 31, 0.87257865942784918, 0.56623880341664246), (2, 9, 0.82805419803149771, 0.53734573174746858), (9, 19, 0.86865475707205908, 0.47620361143286694)], '\xe7\x9b\xb8\xe4\xbc\xbc\xe5\xba\xa6': 0.90869148752794371, '\xe6\xa1\x88\xe7\x94\xb1': u'\u627f\u63fd\u5408\u540c\u7ea0\u7eb7', 'prob': '0.126953'}, {'DM': '9265', 'id': 1277, 'DATASYM': [(1, 3, 0.97455019276159194, 0.56076414389252838), (7, 18, 0.94927755849898887, 0.55057791914219989), (8, 30, 0.93861518573165104, 0.54439377736111549), (4, 6, 0.92539080939811125, 0.53672368178318297), (11, 32, 0.91624062613784563, 0.52721234281407081), (13, 25, 0.88917533341883237, 0.51571882263856761), (2, 4, 0.84894830680914712, 0.49238727707979624), (9, 13, 0.82525503951723178, 0.47864525854531237)], '\xe7\x9b\xb8\xe4\xbc\xbc\xe5\xba\xa6': 0.90843163153417505, '\xe6\xa1\x88\xe7\x94\xb1': u'\u670d\u52a1\u5408\u540c\u7ea0\u7eb7', 'prob': '0.738281'}, {'DM': '9200', 'id': 2548, 'DATASYM': [(7, 8, 0.9758221574446061, 0.6332361728181467), (12, 30, 0.95245105965392174, 0.61807006452000446), (1, 10, 0.93838809408074486, 0.60894424335464792), (10, 28, 0.92341753755057177, 0.59922946300271418), (6, 25, 0.87625718853578127, 0.56862589585570633), (4, 32, 0.86945326384133992, 0.56421065358970413), (2, 22, 0.8602310191103496, 0.5582261010626971), (9, 27, 0.84211310119560201, 0.54646891671077336)], '\xe7\x9b\xb8\xe4\xbc\xbc\xe5\xba\xa6': 0.90476667767661456, '\xe6\xa1\x88\xe7\x94\xb1': u'\u627f\u63fd\u5408\u540c\u7ea0\u7eb7', 'prob': '0.126953'}, {'DM': '9200', 'id': 3767, 'DATASYM': [(12, 32, 0.96093198692649873, 0.62357355702328943), (7, 8, 0.95250733136049959, 0.61810658068213453), (1, 13, 0.94541859780104442, 0.61350651859595284), (10, 31, 0.91412536340507766, 0.59319953147455406), (4, 24, 0.91175959263469497, 0.59166432178806239), (2, 5, 0.85598390287003123, 0.55547003776467174), (3, 4, 0.85300508968580313, 0.55353700904019543), (0, 0, 0.84251411622473427, 0.54672914570880649)], '\xe7\x9b\xb8\xe4\xbc\xbc\xe5\xba\xa6': 0.90453074761354801, '\xe6\xa1\x88\xe7\x94\xb1': u'\u627f\u63fd\u5408\u540c\u7ea0\u7eb7', 'prob': '0.126953'}, {'DM': '9200', 'id': 11674, 'DATASYM': [(12, 11, 0.95185993251557233, 0.61768646686967288), (8, 3, 0.93291213499929448, 0.6053907522345785), (10, 7, 0.92467321165426664, 0.60004430232324446), (6, 9, 0.86063617105616874, 0.55848901461269496), (2, 0, 0.81298835151010485, 0.52756911526188321), (7, 4, 0.962547950409681, 0.19741716179064528), (4, 5, 0.87805498536077931, 0.18008777955661648), (1, 1, 0.90989023953636938, 0.059795547837586227)], '\xe7\x9b\xb8\xe4\xbc\xbc\xe5\xba\xa6': 0.90419537213027956, '\xe6\xa1\x88\xe7\x94\xb1': u'\u627f\u63fd\u5408\u540c\u7ea0\u7eb7', 'prob': '0.126953'}, {'DM': '9265', 'id': 4439, 'DATASYM': [(1, 5, 0.9705897786659311, 0.56293893803152184), (7, 15, 0.94308716614600374, 0.54698751156344116), (4, 10, 0.93586323035895413, 0.5427976521297686), (3, 20, 0.91620666438661036, 0.5313969073279563), (8, 23, 0.90096988596598382, 0.52255962503664821), (2, 4, 0.85856300520372997, 0.49796377110714862), (10, 25, 0.86227305945905874, 0.49615896398210307), (5, 28, 0.84278529426425308, 0.4849455446559246)], '\xe7\x9b\xb8\xe4\xbc\xbc\xe5\xba\xa6': 0.90379226055631567, '\xe6\xa1\x88\xe7\x94\xb1': u'\u670d\u52a1\u5408\u540c\u7ea0\u7eb7', 'prob': '0.738281'}, {'DM': '9265', 'id': 981, 'DATASYM': [(1, 1, 0.96710779783391487, 0.5564817290398848), (7, 8, 0.96644213093373021, 0.55609869886640351), (12, 17, 0.95450766845476842, 0.54923151164039918), (11, 16, 0.92361490523690382, 0.53569366310327271), (4, 3, 0.89632625494869467, 0.51986633403880145), (9, 12, 0.86865475707205908, 0.50381695460908427), (6, 6, 0.8402946445452979, 0.48736818090551903), (10, 13, 0.80488698954030891, 0.4631385504535237)], '\xe7\x9b\xb8\xe4\xbc\xbc\xe5\xba\xa6': 0.9027293935707098, '\xe6\xa1\x88\xe7\x94\xb1': u'\u670d\u52a1\u5408\u540c\u7ea0\u7eb7', 'prob': '0.738281'}]
#
# for i in a:
#     print i
#     # print len(i)
#     # for k, v in i.items():
#     #     print k, v
#
# print '---------------'
#
# a.sort(key=lambda s:s['prob'], reverse=True)
#
# for i in a:
#     print i

# match = re.compile(r'^\s*\n*$')
# if not re.compile(r'^\s*\n*$').match(' \n'):
#     print re.compile(r'^\s*\n*$').match('\n')
# else:
#     print '$$$'


# selectedDocs = "9030	3356	0.9193517203316801	[7 19 0.9561018657099921, 4 8 0.9268517483606157, " \
#                "7 8 0.922309729752997, 4 19 0.9207853763349708, 7 5 0.9149242300452198, 12 19 0.9142866212664799, " \
#                "4 5 0.9109782347013545, 1 8 0.8885759564818119]"
#
# doc_val = selectedDocs.split("\t")
# result_doc_list = []
# if len(doc_val) < 2:
#     print '2'
# newDoc = {}
# newDoc["DM"] = doc_val[0]
# newDoc["id"] = doc_val[1]
#
# newDoc["相似度"] = doc_val[2]
#
# list_row_col = []
# target_list = [x for x in re.compile(r',').split(doc_val[3][1:len(doc_val[3]) - 1])]
# for i in range(len(target_list)):
#     one_row_col = [x for x in re.compile(r' ').split(target_list[i])]
#     if i == 0:
#         list_row_col.append((int(one_row_col[0]), int(one_row_col[1])))
#     else:
#         list_row_col.append((int(one_row_col[1]), int(one_row_col[2])))
#
# newDoc['RowCol'] = list_row_col
#
# result_doc_list.append(newDoc)
#
# for i in range(len(result_doc_list)):
#     print  result_doc_list[i]["DM"], '---', result_doc_list[i]["id"], result_doc_list[i]["相似度"]

# a = {'0': 0.1, '2': 2, '3':-1};
# b = sorted(a.items(), key=lambda k: k[1])

# import sys

# a = sys.argv
# print type(a)
# print type(a[1])

import jieba
import jieba.posseg as pseg

# anyou_dict = '../data/anyou.dic'
#
# jieba.load_userdict(anyou_dict)
#
# sentence = '原告伍某甲、伍某乙、伍某丙起诉称，伍某丁是伤残军人，1948年参加解放革命，1951年参加朝鲜战争，1955年复员安置到广东' \
#            '省云浮县（市）粮食系统下属云城粮所企业工作。当时国家住房政策是实物福利分房制度，因被告客观原因导致伍某丁未享受被' \
#            '告实施住房实物分房待遇。伍某丁1983年离休，离休时与被告产生35年工龄劳动关系，职务为副科。1998年国发（1998）23号国' \
#            '务院《关于进一步深化城镇住房制度改革加快住房建设的通知》规定，停止住房实物分配，逐步实现住房分配货币化。伍某丁未' \
#            '享受被告住房实物分房待遇，所以依照国家有关政策规定被告应支付住房补贴。由于该政策涉及原告切身利益，2003年伍某丁就' \
#            '该侵权事件向被告表示异议。直至2009年9月伍某丁死亡，住房补贴问题一直未能得到解决。伍某丁死亡后其亲属接替维权。2010' \
#            '年1月29日原告与被告通过信访三级程序处理问题。虽然原告主张相关抗辩事实，但未能得到有关部门支持，最后，信访三级程序' \
#            '终结。2014年4月28日为解决纠纷原告申请劳动仲裁遭以超时效不受理。2014年5月19日原告申请行政复议不受理，2014年6月3日' \
#            '原告前往云浮市中级人民法院立案庭提起行政诉讼时，口头告知不受理。 原告认为，云城区人民政府及云城区劳动仲裁委员会不受' \
#            '理原告行政复议及劳动仲裁处理不当。首先，原告住房补贴是国家政策规定被告给予原告支付房屋购房款，其性质属于不动产范' \
#            '畴。依照行政诉讼法规定不动产诉讼时效应为20年。其次，被告对原告作出信访答复意见是被告行政行为，其主体适格，对原告的' \
#            '权利义务产生实际影响，具可复议诉性和可诉性。 本案被告出具证据主张伍某丁1984年及1985年共领取人民币2000元，认定为伍' \
#            '某丁享受国家住房优惠政策待遇，并指建造地址为云城区三元里1号房屋。但该房屋所有权证明，房屋所有权内容没有伍某丁任何' \
#            '信息。该房屋反映1979年拆旧建新，与被告主张伍某丁1984年及1985年相关涉及（公助私建）活动在时间不符。同时原告承认伍' \
#            '某丁1984年及1985年两年间共领取被告2000元用于该房屋外墙灰沙，但因房屋修善领取2000元是正常企业开支活动，属于企事业' \
#            '单位经费范畴，其性质与住房补贴待遇政策无关。 本案被告主张政策文件是地方性法规，属性为下位法。而原告主张国发' \
#            '（1998）23号政策文件为行政法规，属性为上位法。被告引用《云浮市城镇住房货币分配》（云府（1999）25号）文件为依据，' \
#            '否决原告享受（1998）23号住房补贴政策待遇，依照下位法不得抵触上位法相关法律原则，被告适用法律错误。本案被告不按国' \
#            '家有关政策规定向原告支付住房补贴，违反《中华人民共和国劳动合同法》第八十五条未及时足额支付劳动者劳动报酬以及' \
#            '《最高人民法院关于审理劳动争议案件适用法律若干问题的解释》第一条未按照国家有关政策规定给付劳动者住房补贴、' \
#            '住房公积金等福利待遇的相关法律。基于被告当年实施住房补贴分配发放方案信息不公开，伍某丁也无法知道当年所享受住房' \
#            '补贴数额明细。原告提供昭通市巧家县2011年实施离休干部住房补贴政策文件信息，该县离休干部一次住房补贴分配支付发放购' \
#            '房款为人民币340906元，本案伍某丁身份条件性质与巧家县实施住房补贴政策离休干部性质相同。所以原告要求被告按此规定向原' \
#            '告支付一次住房补贴购房款340906元。 综上所述，被告不按国家政策规定向伍某丁支付住房补贴，其行为构成对原告侵权。' \
#            '现向法院起诉请求：1、判令被告依照国家有关政策规定向原告支付一次性住房补贴人民币叁拾肆万零玖佰零陆元（340906元）' \
#            '；2、本案诉讼费由被告承担。'
#
# sentence2 = '原告诉称：多年来我的供暖费一直由被告支付，但被告后来却无端停付。根据北京市有关规定，单位仍然应该负担，现' \
#            '我已自行交纳了2013年至2014年度的供暖费1539．6元，故我诉至北京市高级人民法院，要求判令被告负担2013年至2014年的' \
#            '供暖费1539．6元。'
#
# words = pseg.cut(sentence2)
# for word, flag in words:
#     print word, flag,

# word = '\n\n'
#
# print word.split()
# print ''.join(word.split())

# file_obj_deal = open('../dataTemp/wxjsz/wxjsz.arff', 'w')
#
# file_obj = '../dataTemp/wxjsz.csv'
# file_o = open(file_obj)
# line = file_o.readline()
# while line:
#     list_line = line.strip(' \n').split(',')
#     data = []
#     if list_line[64] == '江苏':
#         data = list_line[0:58]
#         data.append(list_line[59])
#         # data.append(list_line[64])
#         file_obj_deal.write(','.join(data))
#         file_obj_deal.write('\r\n')
#     line = file_o.readline()
# file_o.close()
# file_obj_deal.close()

from matrixText.matrix_seg import list_file
import string


# patt_time = re.compile(r'^(((\d\s?){4}年(\d\s?){1,2}月(\d\s?){1,2}日)|((\d\s?){4}年(\d\s?){1,2}月)|((\d\s?){4}年)|'
#                        r'((\d\s?){1,2}月(\d\s?){1,2}日)|((\d\s?){1,2}日))(低|初|末|以来|年低|年初|年末|月低|月初|月末|开始|期间|份)?$')
#
# patt_time_ymd = re.compile(r'((\d(\s?)){4}年(\d(\s?)){1,2}月(\d(\s?)){1,2}日)(低|初|末|以来)?(\，)')
#
# patt_time_ym = re.compile(r'((\d\s?){4}年(\d\s?){1,2}月)(低|初|末|以来|月低|月初|月末)?(\，)')
#
# patt_time_y = re.compile(r'((\d\s?){4}年)(低|初|末|以来|年低|年初|年末)?(\，)')
#
# patt_time_md = re.compile(r'((\d\s?){1,2}月(\d\s?){1,2}日)(低|初|末|以来)?(\，)')
#
# patt_time_d = re.compile(r'((\d\s?){1,2}日)(低|初|末|以来)?(\，)')
#
# patt_num = re.compile(r'\d+\，\d+')
#
# patt_sym = re.compile(r'\，\w+[\。\，]')
#
#
#
# patt_del = re.compile(r'((\d+\,\d))')
#
# file_lim = '../dataTemp/Temp4.tmp'
# file_lim2 = '../dataTemp/Temp5.tmp'
# file_put = open(file_lim)
# file_put2 = open(file_lim2, 'w')
#
# line = file_put.readline()
# while line:
#     list_line = line.strip(' \r\n').split()
#     # if len(list_line) >= 2 and re.compile('\d+').search(list_line[2]):
#     #     file_put2.write(line)
#     if len(list_line) >= 2 and re.compile('\d+').search(list_line[2]):
#         file_put2.write(line)
#     line = file_put.readline()
#
# file_put.close()
# time.sleep(0)
#
# file_put2.close()
# time.sleep(0)


# print patt_time.match('2013年5月4日')
# print patt_num.search('123，456')
# print patt_sym.search('。200，')

test = '原告，新华，人寿保险股份有限公司上海分公司诉称，2013年1月18日低，原告与被告乔美娜签订《个人业务保险营销员委托合同》（以下简称' \
       '“委托合同”），2014年2月月低，原告委托被告乔美娜在委托合同约定的授权范围内开展保险营销活动，2015年年低，原告按照委托合同约定向被告乔美娜支付佣金。' \
       '《委托合同》第十七条第十九项规定，2016年3月28日以来，被告乔美娜代替或者唆使他人代替客户签署公司规定需客户本人亲笔签名的相关保险资料、' \
       '保险单证或其他重要文件，原告有权单方面解除委托合同，追究被告乔美娜的相关责任，包括被告乔美娜承担一定数额违约金。' \
       '《委托合同》第二十七条规定，在委托合同有效期内，被告乔美娜违反本合同的约定，造成原告经济或声誉损失，原告有权要求被' \
       '告乔美娜赔偿损失。《委托合同》第十二条第四款规定，无论委托合同是否解除或终止，如因被告乔美娜的过错导致保险合同解除' \
       '或无效，被告乔美娜无权获得手续费（佣金），已经领取手续费（佣金）的，应及时退回原告。 2013年3月25日被告乔美娜误导投' \
       '保人孙志渭，在《个人业务投保书》（编号XXXXXXXXXXXXXX）被保险人（法定监护人）签名栏处，让孙志渭代替被保险人吴颖签名' \
       '。经过司法鉴定科学技术研究所司法鉴定中心鉴定并出具鉴定意见书（司鉴中心（2013）技鉴字第1106号），《个人业务投保书》' \
       '被保险人吴颖签名是投保人孙志渭所写。2013年12月2日投保人孙志渭申请退保。2013年12月9日原告退还孙志渭保费20，370元和笔' \
       '迹鉴定费2，000元，申请日现金价值为9，918．3元，退保差额为10，451．7元。2013年3月12日被告乔美娜误导投保人龙静仪，' \
       '在《个人业务投保书》（编号XXXXXXXXXXXXXX）被保险人（法定监护人）签名栏处，让龙静仪代替被保险人叶嘉翔签名。经过，司法鉴' \
       '定科学技术研究所司法鉴定中心鉴定并出具鉴定意见书（司鉴中心（2013）技鉴字第1243号），《个人业务投保书》被保险人叶嘉翔' \
       '签名是投保人龙静仪所写。2013年12月23日投保人龙静仪申请退保。2013年12月27日原告退还龙静仪保费和笔迹鉴定费共计31，' \
       '278元（其中鉴定费为2，000元），申请日现金价值为12，989．1元，退保差额为16，590．9元。2013年3月25日被告乔美娜误导投' \
       '保人傅树琴，在《个人业务投保书》（编号XXXXXXXXXXXXXX、XXXXXXXXXXXXXX）被保险人（法定监护人）签名栏处，让傅树琴或他人' \
       '代替被保险人陈若愚签名。经过司法鉴定科学技术研究所司法鉴定中心鉴定并出具鉴定意见书（司鉴中心（2014）技鉴字第211号），' \
       '《个人业务投保书》被保险人陈若愚签名不是陈若愚本人所写。2014年3月31日投保人傅树琴申请退保。2014年4月8日原告退还傅树琴' \
       '保费共计39，934元，申请日现金价值为16，080．78元，退保差额为23，853．22元。原告支付了笔迹鉴定费3，000元。2013年3月' \
       '原告支付被告上述四份保险合同的佣金共计14，381．44元。鉴于，被告乔美娜的违约行为，给原告造成了经济损失，故，请求：1、判令' \
       '被告赔偿原告保险合同（编号分别为XXXXXXXXXXXX、XXXXXXXXXXXX、XXXXXXXXXXXX、XXXXXXXXXXXX）退保差额损失50，895．82元' \
       '（10，451．7＋16，590．9＋11，926．61＊2）；2、判令被告赔偿原告笔迹鉴定费（保险合同编号分别为XXXXXXXXXXXX、XXXXXXX' \
       'XXXXX、XXXXXXXXXXXX、XXXXXXXXXXXX）7，000元（2，000＋2，000＋3，000）；3、判令被告返还佣金14，381．44元； 4、判令' \
       '诉讼费用由被告承担。后原告撤回了第三项诉讼请求。柳州银行于2012年1月19日为被告华金公司开出票号为：24770301。24770302。' \
       '24770303，24770304。24770305，24770306。24770307，24770308。24770309，24770310，247703ll。24770312，24770313。' \
       '24770314，24770315。24770316，24770317。247703l8，24770319的20张承兑汇票，2017年11月11日低'

test2 = '原告新华人寿保险股份有限公司上海分公司诉称：2013年1月18日低，原告与被告乔美娜签订《个人业务保险营销员委托合同》（以下简称' \
       '“委托合同”），2014年2月月低，原告委托被告乔美娜在委托合同约定的授权范围内开展保险营销活动，2015年年低，原告按照委托合同约定向被告乔美娜支付佣金。' \
       '《委托合同》第十七条第十九项规定，2016年3月28日以来，应及时退回原告。 2013年3月25日被告乔美娜误导投' \
       '保人孙志渭，故请求：1、判令' \
       '被告赔偿原告保险合同（编号分别为XXXXXXXXXXXX、XXXXXXXXXXXX、XXXXXXXXXXXX、XXXXXXXXXXXX）退保差额损失50，895．82元' \
       '（10，451．7＋16，590．9＋11，926．61＊2）；2、判令被告赔偿原告笔迹鉴定费（保险合同编号分别为XXXXXXXXXXXX、XXXXXXX' \
       'XXXXX、XXXXXXXXXXXX、XXXXXXXXXXXX）7，000元（2，000＋2，000＋3，000）；3、判令被告返还佣金14，381．44元； 4、判令' \
       '诉讼费用由被告承担。后原告撤回了第三项诉讼请求。柳州银行于2012年1月19日为被告华金公司开出票号为：24770301。24770302。' \
       '24770303，24770304。24770305，24770306。24770307，24770308。24770309，24770310，247703ll。24770312，24770313。' \
       '24770314，24770315。24770316，24770317。247703l8，24770319的20张承兑汇票, 2017年11月11日低'

test3 = '原告苟京陕诉称： 翟健是三河市超越金海健身中心（系个体工商户）的业主。2013年9月，翟健在北京市顺义区×镇开办了名为魅力' \
        '阳光健身会所（未注册）的健身中心。当时该健身中心尚在装修和招募会员阶段，翟健向原告介绍称该健身中心正在办理营业执照，' \
        '最晚将于2013年年底开业。2013年10月2日，原告支付了2300元的入会费用（含1元办卡费），并由魅力阳光健身会所出具了入会协议' \
        '。 2013年底，该中心迟迟没有正式对外营业，被告告知原告是因为被告内部投资人之间出现纠纷，并承诺会尽快营业。时至今日，' \
        '该中心仍没有正式营业，为此给原告造成了一定的经济损失。 期间，经原告了解，有数十名与原告经历相同的会员在该中心入会并支' \
        '付了金额不等的会费，至今均无法享受健身服务。该中心涉及的会员人数众多，经多次报警及新闻媒体报道，被告均向警方及媒体表示' \
        '该健身中心确实没有正常开业，但由于会员入会款项已经投入到店里，无法退还。原告认为，双方之间形成的服务合同真实有效。被' \
        '告在收取了原告的会费后未在合理期限内开业经营，导致原告入会的目的落空，具有全部过错。根据《合同法》的规定，该服务合同' \
        '应予解除，同时被告应向原告退还全部费用。现原告根据《民事诉讼法》的有关规定提起诉讼，要求：1．判令解除原、被告双方于' \
        '2013年10月2日达成的入会协议；2．判令被告向原告退还会员费2300元；3．判令被告承担本案诉讼费。'


test4 = '原告上海世邦魏理仕物业顾问有限公司诉称，2013年4月，原告与被告签订《太原［联盛金融中心］设计阶段综合顾问服务合同》' \
        '一份，约定原告为被告的太原联盛金融中心项目提供设计阶段综合顾问服务。合同签订后原告依约履行了自身义务，相关报告于2013' \
        '年8月已全部提交，但被告未能及时足额付款。2014年2月24日，被告向原告发函，承诺支付欠款1，580，000元（实际应为1，480' \
        '，000元），3月31日前支付750000元，余款于4月30日前付清。但被告至今仍拖欠原告1，480，000元未支付。特诉至法院，请求依' \
        '法判令：1、被告支付原告顾问费用1480000元；2、被告按日万分之五支付原告自2013年9月1日起至判决生效日止的违约金，违约金' \
        '暂计至2014年8月1日为247160元；3、被告承担本案诉讼费用。'


# a = '（10451．7＋16，590．9＋11，926．61＊2）'
# a2 = '（10，451．7＋16，590．9'
# a3 = '（10451．7'
# a4 = '（，．'
# b = patt_num.search(a)
# b2 = patt_num.search(a2)
# b3 = patt_num.search(a3)
# print len(a4)

# b_all = patt_num.findall(test)
# print len(b_all)
# print len(test)
# for b_one in b_all:
#     test = test.replace(b_one, b_one.replace('，', ','))
# print len(test)
# b_all_re = patt_num.findall(test)
# print len(b_all_re)
#
#
# b_all = patt_sym.findall(test)
# print len(b_all)
# print len(test)
# for b_one in b_all:
#     test = test.replace(b_one, b_one.replace('，', ',', 1))
# print len(test)
# b_all_re = patt_num.findall(test)
# print len(b_all_re)



# match = re.compile(r'\。|\？|\！|\，')
# res_segment = match.split(test)
# print '-----: ', len(res_segment)
#
# res_seg_len = len(res_segment)
# for i in range(res_seg_len):
#     if i > res_seg_len - 1:
#         break
#     if patt_time.match(res_segment[i]):
#         if len(res_segment) == 1:
#             break
#         if i < len(res_segment) - 1:
#             res_segment[i] += ',' + res_segment[i+1]
#             res_segment.remove(res_segment[i+1])
#         else:
#             res_segment[i - 1] += ',' + res_segment[i]
#             res_segment.remove(res_segment[i])
#
#         res_seg_len = len(res_segment)
#
# print '+++++: ', len(res_segment)

# content = '2010年10月14日'
# content2 = '2010年1月14日'
# content3 = '2010年10月'
# content4 = '2010年1月'
# content5 = '10月14日'
# content6 = '10月1日'
# content7 = '14日'
# content8 = '4日'
#
# print pattern.match(content)
# print pattern.match(content2)
# print pattern.match(content3)
# print pattern.match(content4)
# print pattern.match(content5)
# print pattern.match(content6)
# print pattern.match(content7)
# print pattern.match(content8)

# seg_test = seg_sent_up(test4)

# min_sent_len = len('。') * 3
# seg_path = '../dataTfidf/'
# files = list_file(seg_path)
# file_lim = '../dataTemp/Temp4.tmp'
# file_put = open(file_lim, 'w')
# for file_name in files:
#     file_path = seg_path + file_name
#     print '---file: ' + file_path
#     file_obj = open(file_path)
#     line = file_obj.readline()
#     while line:
#         list_line = seg_sentence(line)
#         for content in list_line:
#             if len(content) <= min_sent_len or patt_time.match(content):
#                 word = ''.join(content.split())
#                 if word == '' or word == '\n' or word == '\n\n':
#                     continue
#                 else:
#                     file_put.write(file_name + ' ------ ' + content + '\r\n')
#         line = file_obj.readline()
#     file_obj.close()
#     time.sleep(0)
#
# file_put.close()
# time.sleep(0)


# words = pseg.cut(test)
# list_word = [word for word, flag in words]\


# from matrixText.matrix_seg import seg_cont_ansj, seg_sentence
#
# test5 = '被告承担本案诉讼费用'
#
# print test5
#
# list_sent = seg_sentence(test5)
#
# union_sent = ''
#
# for i in range(len(list_sent) - 1):
#     union_sent += list_sent[i] + 'ansjseg'
# union_sent += list_sent[len(list_sent) - 1]
#
# # seg_text_jion = '-segment-'.jion(list_sent)
#
# ansj_seg, ansj_seg_filter = seg_cont_ansj(to_seg_content=union_sent, out_filter=2,
#                                           ansj_serve_url='http://172.23.4.9:8080/SegService/servlet/Segment')
#
# seg_vec = ansj_seg.strip(' \r\n').split('ansjseg')
# print 'vector: '
# for seg_line in seg_vec:
#     seg_words = seg_line.split()
#     for word in seg_words:
#         print word
#     print '------------'
#
# seg_tfidf = ansj_seg_filter.strip(' \r\n').split('ansjseg')
# print 'tfidf: '
# for seg_line in seg_tfidf:
#     seg_words = seg_line.split()
#     for word in seg_words:
#         print word
#     print '------------'

# for word in ansj_seg:
#     print word



# a = '9111  9112, 9113'
#
# re_file = re.compile(r'9\d{3}')
#
# b = re.compile(r'\d+').findall(a)

# a = ['1']
# b = ['2']
# c = []
# d = ['1', '2']
#
# if ['1'] == a:
#     print a
#
# import numpy as np
#
# a = np.array([1, 2])
#
# print a


# a = (0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
#      1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
#      0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
#      0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
#      1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
#      0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0,
#      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
#      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1,
#      0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
#      0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
#      0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0,
#      1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
#      0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0,
#      0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
#      0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1,
#      0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#      1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
#      1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0,
#      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
#      0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0,
#      1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
#      0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0,
#      1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0,
#      1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1,
#      1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
#      0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
#      1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
#      1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0,
#      0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
#      0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
#      0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
#      0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
#      0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
#      0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1,
#      0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1,
#      1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
#      0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0,
#      0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0,
#      1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0,
#      1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
#      0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0,
#      0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0,
#      1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
#      0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
#      0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
#      0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0,
#      1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
#      0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
#      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1,
#      0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)
#
# print len(a)

# import itertools
#
# re_ele = re.compile(r'\[.*?\]')
#
# file = '/home/uww/Work/data/text/开设赌场罪—总则.txt'
#
# file_sum = open(file, 'r')
# line_sum = file_sum.readline()
# tuple_count = {}
# while line_sum:
#     result = re_ele.findall(line_sum)
#
#     for elementstr in result:
#         list1 = elementstr.strip('[').strip(']').split(", ")
#         list1 = list(set(list1))
#         if len(list1) > 1:
#             for i in range(2, len(list1) + 1):
#                 iter = itertools.combinations(list1, i)
#                 for tuple in list(iter):
#                     key = []
#                     for j in tuple:
#                         key.append(str(j))
#                     key.sort()
#
#                     str_key = ''
#                     for one_key in key:
#                         str_key += one_key + ' '
#
#                     if str_key in tuple_count.keys():
#                         tuple_count[str_key] = tuple_count[str_key] + 1
#                     else:
#                         tuple_count[str_key] = 1
#
#                     # key = ""
#                     # for j in tuple:
#                     #     key = key + str(j) + " "
#                     # if key in tuple_count.keys():
#                     #     tuple_count[key] = tuple_count[key] + 1
#                     # else:
#                     #     tuple_count[key] = 1
#
#
#     line_sum = file_sum.readline()
#
# file_sum.close()
#
# file_obj1 = '/home/uww/Work/data/text/KaiSheDuChangZz1.txt'
#
# file_obj3 = '/home/uww/Work/data/text/KaiSheDuChangZz3.txt'
#
# write_sum1 = open(file_obj1, 'w')
# write_sum3 = open(file_obj3, 'w')
# for k, v in tuple_count.items():
#     if v >= 3:
#         write_sum3.write((k + ' ' + str(v)).encode('utf-8'))
#         write_sum3.write('\r\n'.encode('utf-8'))
#
#     write_sum1.write((k + ' ' + str(v)).encode('utf-8'))
#     write_sum1.write('\r\n'.encode('utf-8'))
#     # print k, v
# write_sum1.close()
# write_sum3.close()

# print(tuple_count)



# re_ele = re.compile(r'^\d+\s\d+\s\d+\s')
#
# file_path = '/home/uww/Work/data/MLData/quora_duplicate_questions.tsv'
# file_ml = '/home/uww/Work/data/MLData/duplicate40000.ml.mini'
#
# file_obj = open(file_path, 'r')
# file_dl = open(file_ml, 'w')
# line = file_obj.readline()
# i = 1
# while line:
#     if i > 40000:
#         break
#     # re_ele.findall(line)
#     list_doc = line.strip(' \r\n').split('\t')
#     file_dl.write((list_doc[3] + '\t' + list_doc[4] + '\t' + list_doc[5] + '\r\n').encode('utf-8'))
#     # print '-----: ', i, ' ---: ', len(list_doc)
#     if i > 1 and (len(list_doc) != 6 or int(list_doc[5]) not in [0, 1]):
#         print '-----: ', i, ' ---: ', len(list_doc)
#     line = file_obj.readline()
#     i += 1
# file_obj.close()
# file_dl.close()




# file_path = '/home/uww/Work/data/MLData/duplicate.ml'
# file_ml = '/home/uww/Work/data/MLData/duplicate.t.learn'
#
# file_obj = open(file_path, 'r')
# file_dl = open(file_ml, 'w')
# line = file_obj.readline()
# line = file_obj.readline()
# i = 1
# while line:
#     # re_ele.findall(line)
#     list_doc = line.strip(' \r\n').split('\t')
#     file_dl.write((list_doc[0] + '\r\n' + list_doc[1] + '\r\n').encode('utf-8'))
#     # print '-----: ', i, ' ---: ', len(list_doc)
#     # if i > 1 and (len(list_doc) != 6 or int(list_doc[5]) not in [0, 1]):
#     #     print '-----: ', i, ' ---: ', len(list_doc)
#     line = file_obj.readline()
#     i += 1
# file_obj.close()
# file_dl.close()

# file_path = '/home/uww/Work/data/MLData/temp.tmp'
#
# file_obj = open(file_path, 'r')
# line = file_obj.readline()
# list_doc = line.strip(' \r\n').split('\t')
# file_obj.close()



# tmp_dict = dict()
#
# tmp_dict['a'] = 1
# tmp_dict['b'] = 2
#
# for k, v in tmp_dict.items():
#     print k, v
#
# print '------'
#
# tmp_list = [('a', 1), ('b', 2)]
#
# for k, v in tmp_list:
#     print k, v
#
#
# print 'ok'


# a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
#      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
#      31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
#
# dim = 8
# a = [a[i * dim: (i + 1) * dim] for i in range(len(a) / dim)]
#
# for param in a:
#     print param
#
# print len(a) % 7 == 0
# print len(a) % 5 == 0
#
# print '------'
# print a
#
# a = np.nan
#
# print type(a), a
# print type(float(a)), float(a)

# a = [(0, 0, 0.9370830560296377), (25, 4, 0.8115020223650347), (11, 3, 0.8042089776896653), (32, 5, 0.7941603017298694),
#      (1, 2, 0.6611980127214796), (31, 1, 0.5472209066026755)]
#
# b = np.sum(a, axis=0)
#
# b /= len(a)
# print b, np.mean(a, axis=0)

# a = [(
#      u'0.081831 -0.029801 0.0095816 -0.058729 -0.029493 0.079813 -0.022438 -0.024154 0.03371 0.045731 -0.039015 0.087481 -0.053553 0.097174 -0.056476 -0.01945 0.042813 -0.041397 0.066587 -0.061025 0.05495 0.087022 0.072281 -0.047458 -0.035438 -0.045221 0.022958 -0.11777 0.013548 0.074149 -0.056169 0.025565 -0.0022459 -0.016746 -0.17114 -0.036621 -0.046449 -0.0061251 -0.0058644 -0.016379 -0.077124 0.056456 0.0041112 -0.042943 0.051447 -0.037746 0.045811 -0.025846 0.0015799 0.055578 -0.11947 -0.012944 -0.050388 0.11471 -0.081422 -0.042438 -0.045025 0.053399 0.078736 0.097995 -0.046994 0.081699 0.064462 0.043222 -0.011358 0.10103 -0.050089 -0.024327 -0.0032523 -0.065053 0.0063788 0.0087858 0.050711 -0.006401 0.037607 -0.019105 0.10741 0.072649 0.12189 0.05009 0.091172 0.024476 0.096484 -0.099559 0.13317 -0.057034 0.11272 0.074395 -0.046408 0.011518 0.087626 0.0037046 0.17579 -0.031041 -0.020482 -0.10855 -0.071537 0.061412 -0.054648 -0.048669 ',
#      0), (
#      u'0.05293 0.015024 0.0051384 -0.036644 -0.028147 0.04472 0.010248 -0.015304 0.021713 0.023902 -0.017447 0.062525 -0.038952 0.06116 -0.025951 -0.021094 0.025984 -0.031351 0.05051 -0.065808 0.017518 0.072324 0.051342 -0.023385 -0.027103 -0.014271 0.025677 -0.044813 0.0069906 0.049893 -0.013435 0.00055686 -0.027837 -0.020103 -0.066376 -0.047168 -0.03228 -0.0011965 -0.011273 -0.019067 -0.07151 0.038387 0.0024467 -0.021286 0.04697 -0.043234 0.030807 -0.01797 0.0044407 0.022999 -0.086541 -0.0027647 -0.048533 0.062307 -0.039534 -0.054664 -0.061855 0.031887 0.057387 0.06698 -0.014818 0.047806 0.023992 0.017867 0.027319 0.083427 -0.032533 -0.014865 0.0042743 -0.035626 0.011083 -0.0050789 0.012683 -0.018895 0.0071743 -0.025954 0.064623 0.05241 0.093501 0.038053 0.032763 0.027332 0.052768 -0.069795 0.098694 -0.039215 0.076332 0.033996 -0.019072 0.015815 0.062826 -0.015194 0.11057 0.0038754 -0.006329 -0.05372 -0.033503 0.069526 -0.03759 -0.030836 ',
#      1), (
#      u'0.074326 -0.0009895 0.017085 -0.045168 0.002139 0.065939 -0.0019087 -0.0023613 0.018319 0.04981 -0.029645 0.053615 -0.01291 0.072232 -0.049052 -0.024476 0.023242 -0.023037 0.040754 -0.026163 0.0286 0.064964 0.048229 -0.039614 -0.022793 -0.015836 0.041355 -0.068523 0.011615 0.063737 -0.01204 0.009801 0.0096306 -0.038372 -0.11422 0.0003782 -0.055374 -0.044017 -0.024622 -0.012171 -0.075413 0.049625 -0.025324 -0.0058096 0.019748 -0.024059 -0.010701 -0.037892 0.0059926 0.021864 -0.115 0.035997 -0.06564 0.08277 -0.028225 -0.058476 -0.018724 0.050335 0.069156 0.078677 -0.032762 0.067819 0.054759 0.03428 0.040636 0.084972 -0.05017 -0.01938 0.015115 0.0019738 -0.017238 0.028703 0.038543 -0.0074161 0.046204 -0.018888 0.10942 0.072953 0.10437 0.044691 0.048707 0.037274 0.06982 -0.068682 0.10168 -0.027078 0.061077 0.064837 -0.064543 0.0048092 0.062984 -0.00060673 0.15431 -0.033785 -0.029831 -0.075057 -0.054408 0.051028 -0.056896 -0.033377 ',
#      2), (
#      u'0.060627 0.005922 0.010679 -0.053871 0.0090105 0.053484 0.009582 -0.015233 0.034437 0.049519 -0.024236 0.02807 -0.012778 0.067325 -0.04421 -0.019677 0.019894 -0.026464 0.038249 -0.044103 0.032197 0.034954 0.032892 -0.037452 -0.0076061 -0.021826 0.040797 -0.076359 0.01032 0.066785 -0.034048 0.012091 -0.013302 -0.028614 -0.097139 -0.024984 -0.058152 -0.014674 -0.0064358 -0.0022983 -0.064239 0.038583 -0.0022498 -0.022427 0.030326 -0.026207 0.015842 -0.028521 0.0096718 0.033384 -0.10743 0.0079851 -0.061164 0.099949 -0.051005 -0.052361 -0.031057 0.052465 0.070334 0.081948 -0.039468 0.0731 0.061837 0.038208 0.031377 0.095089 -0.054759 -0.03157 0.0096555 -0.038399 -0.0010535 0.015515 0.03351 -0.01835 0.02382 -0.024658 0.091212 0.060605 0.10329 0.024516 0.060682 0.028112 0.073988 -0.096091 0.096311 -0.04107 0.076591 0.065825 -0.040836 0.004948 0.06996 -0.0013432 0.14398 -0.035287 -0.014144 -0.081956 -0.053229 0.038999 -0.049988 -0.042076 ',
#      3), (
#      u'0.10609 0.02602 0.052934 -0.028665 0.04032 0.07682 -0.019163 0.064885 -0.030252 -0.0032486 -0.062487 0.039788 0.038123 0.042692 -0.033742 -0.016202 0.030232 0.018939 0.038373 0.048748 0.0235 -0.014627 0.027118 -0.054682 -0.01555 0.031146 0.047089 0.027042 0.061739 0.021247 0.069924 -0.01184 0.053589 -0.030207 -0.097743 0.0018577 0.066568 0.019929 -0.025559 -0.02216 -0.083835 0.054565 -0.0060101 0.041857 0.092834 -0.083904 0.049898 0.0070754 -0.026165 -0.01459 -0.049249 0.047507 0.013616 0.017526 0.00064548 -0.028292 -0.010434 0.0032025 0.034437 0.013872 0.001203 0.035712 -0.030066 0.0092429 0.019094 -0.080234 0.025418 0.042189 -0.010325 0.0099827 -0.051587 0.035989 0.0030928 0.083504 0.0226 0.016377 0.038229 0.024972 0.025759 0.070871 -0.017513 0.022989 0.013017 0.019665 0.0018824 0.091448 0.042635 0.064256 -0.023781 -0.035575 -0.014904 0.0079359 0.048324 -0.038348 -0.085239 -0.061676 -0.068939 -0.044071 -0.016198 -0.096955 ',
#      4), (
#      u'0.09815 -0.022828 0.010354 -0.041859 -0.045569 0.082049 0.041305 0.0013947 0.012112 0.026507 -0.0015594 0.060424 -0.00052806 0.089253 -0.04587 -0.03084 0.028829 -0.021469 0.056627 -0.044317 -0.004562 0.043356 0.001329 -0.020662 0.0050206 0.0063406 0.059891 -0.035768 0.040137 0.066903 -0.0012571 -0.0098353 -0.0037178 -0.035786 -0.088207 -0.0216 -0.031816 0.039948 -0.0040762 0.0049741 -0.14324 0.024227 -0.038012 -0.0013586 0.026154 -0.046057 -0.010509 -0.044834 0.026713 -0.018616 -0.13874 0.034337 -0.076879 0.070925 -0.012016 -0.088674 -0.041062 0.068172 0.058168 0.10621 -0.02096 0.044779 0.01506 0.010606 0.060798 0.13725 -0.040791 -0.056487 -0.0060529 -0.058257 0.035226 -0.0012384 0.035627 -0.00032472 0.075835 -0.048118 0.10622 0.075814 0.10479 0.07036 0.03168 0.043149 0.067879 -0.055923 0.11098 -0.022946 0.089495 0.060454 -0.017461 0.0063387 0.062921 -0.013012 0.14856 0.0053708 -0.039755 -0.050571 -0.04024 0.06515 -0.052779 -0.0053101 ',
#      5), (
#      u'0.063785 0.013642 0.010274 -0.037264 -0.017939 0.045477 0.011737 -0.0078331 0.018647 0.02642 -0.023741 0.063181 -0.035202 0.059778 -0.02877 -0.024641 0.025633 -0.027012 0.051276 -0.060545 0.016159 0.072405 0.055996 -0.028162 -0.028582 -0.0077898 0.032829 -0.042044 0.015426 0.05408 -0.0054589 -0.0028721 -0.017488 -0.027725 -0.073029 -0.039735 -0.035617 -0.011552 -0.018958 -0.019149 -0.075065 0.04154 0.00073142 -0.012276 0.047212 -0.04612 0.025233 -0.019486 0.0017459 0.025214 -0.088547 0.0066612 -0.055058 0.059201 -0.027543 -0.061567 -0.05667 0.035057 0.060601 0.069963 -0.015459 0.050997 0.02221 0.018837 0.035162 0.084933 -0.035173 -0.01576 0.0053186 -0.022115 0.0060666 0.0022313 0.016255 -0.014431 0.015748 -0.022046 0.073023 0.057917 0.096684 0.043734 0.028877 0.032305 0.052953 -0.065678 0.099 -0.034194 0.072427 0.038963 -0.024539 0.013176 0.060247 -0.016103 0.12002 0.0031724 -0.015865 -0.05029 -0.037532 0.067336 -0.043617 -0.030217 ',
#      6), (
#      u'0.084787 -0.033601 0.0222 -0.025659 -0.011013 0.055695 0.034028 0.013201 0.003422 0.021771 -0.011087 0.045187 0.0031762 0.065486 -0.04118 -0.028393 0.021516 -0.0099407 0.04174 -0.020514 -0.0073904 0.040216 0.017215 -0.022605 -0.0040054 0.010702 0.05131 -0.028095 0.043816 0.053294 0.0075395 -0.012255 0.020793 -0.03831 -0.081237 0.00095774 -0.034398 0.0044498 -0.019812 0.0004856 -0.088947 0.020996 -0.024887 0.0095087 0.0053843 -0.02381 -0.021283 -0.027832 0.012273 -0.0023384 -0.082769 0.033564 -0.066388 0.047805 0.015365 -0.079951 -0.013984 0.056049 0.045274 0.077818 -0.016921 0.034194 0.006542 0.0079416 0.053951 0.10843 -0.03421 -0.051455 -0.0021691 -0.016825 0.016943 0.010815 0.033275 0.0069819 0.072933 -0.029744 0.09194 0.06634 0.082276 0.062956 0.014881 0.041746 0.050625 -0.023242 0.073538 -0.013372 0.043896 0.041046 -0.016454 0.0011171 0.038428 -0.01588 0.1223 0.012736 -0.040756 -0.014315 -0.029405 0.047398 -0.05033 0.010662 ',
#      7), (
#      u'0.10534 -0.060445 0.0089956 -0.067 -0.020271 0.077847 -0.032891 -0.016415 0.017012 0.058355 -0.042473 0.12257 -0.065902 0.11585 -0.057553 -0.021464 0.04843 -0.053853 0.081723 -0.075037 0.049214 0.10272 0.12213 -0.051615 -0.066194 -0.03138 0.017883 -0.11353 0.016317 0.076451 -0.048214 0.019694 -0.028908 -0.017701 -0.19437 -0.076913 -0.059315 0.00060408 -0.022908 -0.030825 -0.10743 0.016962 0.023576 -0.056089 0.047296 -0.055606 0.029937 -0.034218 0.018722 0.066572 -0.11858 -0.02254 -0.076344 0.084697 -0.05645 -0.074769 -0.068802 0.055879 0.083891 0.10508 -0.037013 0.067368 0.043539 0.025253 0.009598 0.1576 -0.060735 -0.039062 -0.0027191 -0.02133 -0.0022461 0.016168 0.042027 -0.01356 -0.00024593 0.0017051 0.10429 0.089278 0.16799 0.072587 0.10147 0.023274 0.095434 -0.13582 0.15906 -0.058889 0.11215 0.061345 -0.045982 0.034096 0.089912 -0.010786 0.17272 0.0048876 -0.0097481 -0.060936 -0.072867 0.1109 -0.067295 -0.035865 ',
#      8), (
#      u'0.10088 -0.015139 0.027689 -0.034903 0.023047 0.064921 -0.042334 0.020082 0.020315 0.013162 -0.084551 0.065069 -0.038491 0.04708 -0.055762 -0.014636 0.034433 0.0034359 0.044735 0.020101 0.048732 0.08154 0.064342 -0.054394 -0.058924 -0.0066608 0.034711 -0.050936 0.023106 0.062923 0.012302 0.01012 0.053381 -0.027434 -0.18205 0.0079188 0.020749 -0.0029924 -0.028267 -0.010783 -0.065533 0.042937 0.030976 0.048633 0.093595 -0.079894 0.059547 0.00050298 -0.03085 0.065608 -0.059129 0.0063841 -0.024046 0.050969 -0.048003 -0.031971 -0.025444 0.035316 0.058752 0.069721 -0.022092 0.066669 -0.0052692 0.02779 -0.01879 0.0049917 -0.031034 0.011336 -0.023491 0.023576 -0.036854 0.040528 0.01247 0.04732 -0.015681 0.050461 0.068319 0.03425 0.076579 0.045965 0.043907 0.0045771 0.050321 -0.092617 0.070709 0.009867 0.054695 0.035928 -0.031252 -0.0059284 0.049474 -0.010707 0.11632 -0.01711 -0.050684 -0.082918 -0.062405 0.0029067 -0.030149 -0.062334 ',
#      9)]
#
# iter_a = iter(a)
#
# def make_iter():
#     for iter_i in iter_a:
#         # print type(iter_i)
#         # print iter_i
#         # print iter_i[0]
#         # print iter_i[1]
#         # print '----'
#         yield iter_i[0][0:10], iter_i[1]
#
# b = make_iter()
#
# print type(b), b
# print '======'
#
# for iter_i in b:
#     print type(iter_i)
#     print iter_i
#     print iter_i[0]
#     print iter_i[1]
#     print '----'




'''
######
# map-reduce mp
######

from multiprocessing.dummy import Pool as TreadPool


file_anli = '../../anyou/simimat.test'
file_anli2 = '../../anyou/simimat2.test'

obj_anli = open(file_anli, 'r')
obj_anli2 = open(file_anli2, 'r')

mat_line = obj_anli.readline()
mat_line2 = obj_anli2.readline()

x_split = mat_line.split()
y_split = mat_line2.split()

x_matrix = [float(param) for param in x_split]
y_matrix = [float(param) for param in y_split]

obj_anli.close()
obj_anli2.close()

vec_dim = 100

xi_dim = len(x_matrix) / vec_dim
yi_dim = len(y_matrix) / vec_dim

x_matrix = [x_matrix[i * vec_dim: (i + 1) * vec_dim] for i in range(xi_dim)]
y_matrix = [y_matrix[i * vec_dim: (i + 1) * vec_dim] for i in range(yi_dim)]

def corr_dist(x, y):

    if len(x) != len(y):
        return 0.0

    # avg = (sum(x) + sum(y)) / (len(x) + len(y))
    x_avg = sum(x) / len(x)
    y_avg = sum(y) / len(y)
    x_tran = map(lambda a: a - x_avg, x)
    y_tran = map(lambda b: b - y_avg, y)

    res_fenzi = map(lambda (a, b): a * b, zip(x_tran, y_tran))
    res_fenmu0 = map(lambda a: a ** 2, x_tran)
    res_fenmu1 = map(lambda b: b ** 2, y_tran)

    if sum(res_fenmu0) < 0.0000001 or sum(res_fenmu1) <= 0.000001:
        return 0.0

    return sum(res_fenzi) / (sum(res_fenmu0) * sum(res_fenmu1)) ** 0.5

list_mul = []

start_time = time.time()
# ---mp---

map_xy = []

for i in range(len(x_matrix)):
    for j in range(len(y_matrix)):
        map_xy.append((i, x_matrix[i], j, y_matrix[j]))

print time.time() - start_time

def mul_mat(xy_map):
    # print '------', xy_map[0], xy_map[2], corr_dist(xy_map[1], xy_map[3])
    list_mul.append((xy_map[0], xy_map[2], corr_dist(xy_map[1], xy_map[3])))

pool = TreadPool(4)

res = pool.map(mul_mat, map_xy)

pool.close()
pool.join()

print '------', len(list_mul), list_mul

print 'mp-time: ', time.time() - start_time

# ---mp---


start_time = time.time()
# ---sp---

for i in range(len(x_matrix)):
    for j in range(len(y_matrix)):
        # temp_dist = corr_dist(x_matrix[i], y_matrix[j])
        list_mul.append((i, j, corr_dist(x_matrix[i], y_matrix[j])))

print '------', len(list_mul), list_mul

print 'sp-time: ', time.time() - start_time

# ---sp---

'''


'''
# file-double-write
file_src_mat = os.path.expanduser('~/fastText/anyou/minshi/cbowVectAnsj/9293.txt.out.ansj.vect')
file_dbl_mat = os.path.expanduser('~/fastText/anyou/minshi/cbowVectAnsj/9293w100.txt.out.ansj.vect')

src_mat = codecs.open(file_src_mat, 'r', 'utf-8')
dbl_mat = codecs.open(file_dbl_mat, 'a', 'utf-8')

line = src_mat.readline()

max_seek = 20
seek = 1

while line:
    dbl_mat.write(line.encode('utf-8'))
    dbl_mat.flush()

    line = src_mat.readline()

    if not line and seek < max_seek:
        seek += 1
        src_mat.seek(0)
        line = src_mat.readline()

# dbl_mat.write('\r\n'.encode('utf-8'))
# dbl_mat.flush()

src_mat.close()
dbl_mat.close()
'''



'''
# weizhi Semantics test

from matrixText.matrix_seg import seg_sentence
from matrixText.matrix_multiply import corr_distance

# doc1 = '原告秦皇岛日飞昕虹仪器仪表有限公司诉称，原、被告于2009年5月5日、2009年6月12日分别签订了《技术服务合同》，约定由原告为' \
#        '被告所承包的黄骅港工程提供海上施工作业中的测量、定位服务。合同签订后，原告按约定从2009年5月至12月期间为被告提供了相应' \
#        '技术服务，合计应收费45．3万元。但被告实际给付费用17．2万元，并于2011年期间给付5万元，尚欠23．1万元至今未付，原告就' \
#        '欠款事宜多次找被告催要，被告一直推拖至今。故原告诉至法院，要求判令被告给付服务费人民币23．1万元及利息（自2010年1月1日' \
#        '起至判决生效日止，按中国人民银行同期贷款利率计算）'
#
# doc2 = '原告资阳市力源粮油有限公司诉称，2013年5月31日，原、被告签订《委托担保合同》，被告为原告向中国农业发展银行资阳市分行' \
#        '贷款800万元提供保证担保，被告收取了担保费16万元及担保保证金160万元。2014年5月28日原告归还银行借款800万元，但被告未能' \
#        '及时返还原告保证金，原告多次与被告交涉无果，故诉至法院，请求判令：1．被告退还原告担保保证金160万元并承担从2014年5月29日' \
#        '至返还保证金期间的资金利息（以160万元为基数按每日万分之一点七五计算）；2．被告承担本案诉讼费'

# doc1 = '原告许某诉称，原、被告双方于2011年经人介绍相识并确立恋爱关系，××××年××月××日登记结婚。婚后感情一般，被告不务正业，婚后不久就分居生活至今，期间无任何联系，双方婚后未生育子女。原告曾于2011年10月份、2012年6月份两次向法院起诉要求与被告离婚，均未获法院准许。现原告认为夫妻感情已彻底破裂，故再次诉至法院要求与被告离婚。'
# doc2 = '原告金某诉称，原、被告于2004年相识，××××年××月××日结婚，双方因婚前缺乏了解，婚后由于性格不合经常发生吵闹，2010年原告曾起诉离婚，后被法院驳回其诉讼请求，现再次起诉请求法院判决其与被告离婚。'


# list_mul = list()
#
# res_seg_doc1 = seg_sentence(doc1)
# res_seg_doc2 = seg_sentence(doc2)
#
# sim_senten_len = min(len(res_seg_doc1), len(res_seg_doc2))
#
# for i in range(len(res_seg_doc1)):
#     for j in range(len(res_seg_doc2)):
#         url = 'http://api.yougelvshi.cn/hlsimilarity/divorce?question2=' + res_seg_doc1[i].encode('utf-8') + \
#               '&question1=' + res_seg_doc2[j].encode('utf-8')
#         req = urllib2.Request(url)
#         response = urllib2.urlopen(req)
#         res = response.read()
#         time.sleep(0)
#         list_mul.append((i, j, int(str(res).strip(' \r\n'))))
#
# list_mul.sort(key=itemgetter(2), reverse=True)
#
# list_result = []
# list_col = []
# list_row = []
# for list_one in list_mul:
#     if len(list_result) >= sim_senten_len:
#         break
#     if list_one[0] not in list_row and list_one[1] not in list_col:
#         list_row.append(list_one[0])
#         list_col.append(list_one[1])
#         list_result.append(list_one)
#
# for one in list_result:
#     print one


# for i in range(len(res_seg_doc1)):
#     list_one = list()
#     for j in range(len(res_seg_doc2)):
#         url = 'http://api.yougelvshi.cn/hlsimilarity/divorce?question2=' + res_seg_doc1[i].encode('utf-8') + \
#               '&question1=' + res_seg_doc2[j].encode('utf-8')
#         req = urllib2.Request(url)
#         response = urllib2.urlopen(req)
#         res = response.read()
#         time.sleep(0)
#         list_one.append((i, j, int(str(res).strip(' \r\n'))))
#     if list_one:
#         list_one.sort(key=itemgetter(2), reverse=True)
#         list_mul.append(list_one)
#
# for one in list_mul:
#     print one

# doc1_mat = ''
# doc2_mat = ''
# doc2_weight = '0.595098922947 0.551705456083 0.595098922947 0.551705456083 0.595098922947 0.595098922947 0.595098922947 0.0565492639603 0.595098922947 '

# doc1_mat = ''
# doc2_mat = ''
# doc2_weight = '0.437315322038 0.583040358832 0.0557143073475 0.141240783191 0.234318059847 0.437315322038 0.0613192311513 0.583040358832 '
#
# x_split = doc1_mat.split()
# y_split = doc2_mat.split()
# weight_split = doc2_weight.split()
#
# x_matrix = [float(param) for param in x_split]
# y_matrix = [float(param) for param in y_split]
# weight_matrix = [float(param) for param in weight_split]
#
# vec_dim = 100
#
# print len(x_matrix) % vec_dim == 0
# print len(y_matrix) % vec_dim == 0
#
# xi_dim = len(x_matrix) / vec_dim
# yi_dim = len(y_matrix) / vec_dim
#
# x_matrix = [x_matrix[i * vec_dim: (i + 1) * vec_dim] for i in range(xi_dim)]
# y_matrix = [y_matrix[i * vec_dim: (i + 1) * vec_dim] for i in range(yi_dim)]
#
# sim_senten_len = min(len(x_matrix), len(y_matrix))
#
# list_mul = list()

# for i in range(len(x_matrix)):
#     list_one = list()
#     for j in range(len(y_matrix)):
#         temp_dist = corr_distance(x_matrix[i], y_matrix[j])
#         weight_dist = temp_dist * weight_matrix[j]
#         # temp_dist = Decimal(temp_dist).quantize(Decimal('0.00'))
#         temp_dist = '%.2f' % temp_dist
#         weight_dist = '%.2f' % weight_dist
#         list_one.append((i, j, float(temp_dist), float(weight_dist)))
#
#     if list_one:
#         list_one.sort(key=itemgetter(3), reverse=True)
#         list_mul.append(list_one)
#
# for one in list_mul:
#     print one


# for i in range(len(x_matrix)):
#     for j in range(len(y_matrix)):
#         temp_dist = corr_distance(x_matrix[i], y_matrix[j])
#         weight_dist = temp_dist * weight_matrix[j]
#         temp_dist = '%.2f' % temp_dist
#         weight_dist = '%.2f' % weight_dist
#         list_mul.append((i, j, float(temp_dist), float(weight_dist)))
#
# list_mul.sort(key=itemgetter(3), reverse=True)
#
# list_result = []
# list_col = []
# list_row = []
# mean_dist = 0.0
# for list_one in list_mul:
#     if len(list_result) >= sim_senten_len:
#         break
#     if list_one[0] not in list_row and list_one[1] not in list_col:
#         list_row.append(list_one[0])
#         list_col.append(list_one[1])
#         list_result.append(list_one)
#         mean_dist += list_one[2]
#
# print mean_dist / len(list_result)
# for one in list_result:
#     print one


# file_mat = '/home/uww/fastText/anyou/minshi/cbowVectAnsj/9015.txt.out.ansj.vect'
# file_weight = '/home/uww/fastText/anyou/minshi/tfidfTextAnsj/9015.txt.content.ansj.tfidf'
#
# obj_mat = open(file_mat, 'r')
# obj_weight = open(file_weight, 'r')
#
# line_mat = obj_mat.readline()
# line_weight = obj_weight.readline()
# line_num = 1
#
# while True:
#     if line_num == 33010:
#         print line_mat
#         print line_weight
#         break
#     line_mat = obj_mat.readline()
#     line_weight = obj_weight.readline()
#     line_num += 1
#
# obj_mat.close()
# obj_weight.close()
'''

# student_tuples = [['john', 'A', 15], ['jane', 'B', 16], ['dave', 'B', 10]]
#
# print student_tuples.sort(key=lambda student: student[2], reverse=True)
# print student_tuples

simi_line = '原告许某诉称，原、被告双方于2011年经人介绍相识并确立恋爱关系，××××年××月××日登记结婚。婚后感情一般，被告不务正业，婚后不久就分居生活至今，期间无任何联系，双方婚后未生育子女。原告曾于2011年10月份、2012年6月份两次向法院起诉要求与被告离婚，均未获法院准许。现原告认为夫妻感情已彻底破裂，故再次诉至法院要求与被告离婚'
file_src = '/home/uww/fastText/anyou/minshi/src/9015.txt'

obj_src = open(file_src, 'r')
line_src = obj_src.readline()
line_num = 1

dict_simi = dict()

while line_src:

    # url = 'http://api.yougelvshi.cn/hlsimilarity/divorce?question2=' + simi_line + \
    #       '&question1=' + line_src
    # req = urllib2.Request(url)
    # response = urllib2.urlopen(req)
    # res = response.read()
    # time.sleep(0)
    # dict_simi[line_num] = int(str(res).strip(' \r\n'))

    try:
        results = requests.get("http://api.yougelvshi.cn/hlsimilarity/divorce",
                               params={'question2': simi_line, 'question1': line_src})
    except IOError,e:
        dict_simi[line_num] = 0
        print line_num, e
    else:
        res_simi = results.content
        dict_simi[line_num] = int(str(res_simi).strip(' \r\n'))
        print line_num, ' --> ', str(res_simi).strip(' \r\n')
    finally:
        line_src = obj_src.readline()
        line_num += 1




dict_simi = sorted(dict_simi.iteritems(), key=lambda key_val: key_val[1], reverse=True)

i = 0
for k, v in dict_simi.iteritems():
    if i < 10:
        print k, v
        i += 1
    else:
        break


print 'ok'















