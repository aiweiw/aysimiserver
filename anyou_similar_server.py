#!/usr/bin/env python
# -*- coding:utf-8 -*-
import ConfigParser
import multiprocessing
import urllib
import fasttext

import tornado.ioloop
import tornado.web
import os
import codecs
import jieba
import sys
import re
import node
import time
import urllib2
import jieba.posseg as pseg
import logging
import logging.handlers

if sys.getdefaultencoding() is not 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

work_dir = ''  # similar anyou server work dir
server_port = ''  # similar anyou server port
similog = None  # similar anyou server log
mp_server_url = list()  # java mp server ip
minshi_firstlist = list()  # minshi json data
minshi_nodemap = dict()
minshi_label_map = dict()
xingshi_firstlist = list()  # xingshi json data
xingshi_nodemap = dict()
xingshi_label_map = dict()


class FinalLogger:
    logger = None

    levels = {'n': logging.NOTSET,
              'd': logging.DEBUG,
              'i': logging.INFO,
              'w': logging.WARN,
              'e': logging.ERROR,
              'c': logging.CRITICAL}

    log_level = 'd'
    log_file = 'anyou_classify_logger.log'
    log_max_byte = 10 * 1024 * 1024;
    log_backup_count = 5

    @staticmethod
    def getLogger():
        if FinalLogger.logger is not None:
            return FinalLogger.logger

        FinalLogger.logger = logging.Logger('oggingmodule.FinalLogger')
        log_handler = logging.handlers.RotatingFileHandler(filename=FinalLogger.log_file, \
                                                           maxBytes=FinalLogger.log_max_byte, \
                                                           backupCount=FinalLogger.log_backup_count)
        log_fmt = logging.Formatter('%(asctime)s %(levelname)s %(filename)s[line:%(lineno)d] %(message)s')
        log_handler.setFormatter(log_fmt)
        FinalLogger.logger.addHandler(log_handler)
        FinalLogger.logger.setLevel(FinalLogger.levels.get(FinalLogger.log_level))
        return FinalLogger.logger


def get_all_grandsons_id(child, grandson_list):
    grandson_list.append(child.get('DM'))
    if 'childlist' in child.keys():
        for grand_son in child.get('childlist'):
            get_all_grandsons_id(grand_son, grandson_list)


def segment_str(sentence):
    seg_list = pseg.cut(sentence)
    result = ""
    char_limit = 0
    for char in seg_list:
        if char.flag == 'nr':
            char.word = 'NR'
        result = result + char.word + ' '
        char_limit += 1
    result = re.sub(r'\s{1,}', ' ', result)
    return result.strip().lower()


def clean_str(string):
    seg_list = jieba.cut(string, cut_all=False)
    result = ""
    char_limit = 0
    for char in seg_list:
        if char_limit == 500:
            break
        result = result + char + ' '
        char_limit += 1
    result = re.sub(r'\s{1,}', ' ', result)
    return result.strip().lower()


def callfastText(anyou_hit, model_id, anyou_type, srcfile):
    predictcmd = './fasttext predict-prob  anyou/' + anyou_type + '/bin/' \
                 + str(model_id) + '.bin ' + srcfile + ' ' + str(anyou_hit)
    print(predictcmd)
    cmdresult = os.popen(predictcmd).readlines()
    return cmdresult


def recursiveclassify(curDM, anyou_type, label_map, srcfile):
    result = callfastText(3, curDM, anyou_type, srcfile)
    labels = []
    i = 0
    if len(result) < 1:
        return labels
    testresult = result[0].split(' ')
    while i < len(testresult):
        label_map_key = curDM + testresult[i]
        label = {}
        label['DM'] = label_map.get(label_map_key.rstrip('\n')).get('DM')
        label['anyou_label'] = label_map.get(label_map_key.rstrip('\n')).get('MC')
        label['prob'] = testresult[i + 1]
        if float(label['prob']) < 0.1:
            break
        label['子类'] = recursiveclassify(label['DM'], anyou_type, label_map, srcfile)
        labels.append(label)
        i += 2
        if float(label['prob']) > 0.9:
            break
    return labels


def getLeaves(labelnode, dict_target_DMs):
    if '子类' in labelnode.keys():
        childlist = labelnode['子类']
        if len(childlist) > 0:
            for child in childlist:
                getLeaves(child, dict_target_DMs)

    dict_target_DMs[labelnode['DM']] = float(labelnode['prob'])


def run_mp_server(server_url, server_param):
    data_simi_param = urllib.urlencode(server_param)
    req = urllib2.Request(server_url, data=data_simi_param)
    response = urllib2.urlopen(req)
    selected_docs = response.read()
    selected_docs = str(selected_docs).split(';')
    return selected_docs


def get_simi_docs(src, doc_hit, anyou_type, anyou_type_assign, anyou_id, if_classify):
    start_time = time.time()
    model_basedir = work_dir + 'anyou/'
    if ['assign'] == if_classify and anyou_id:
        targetDMs = re.compile(r'\d+').findall(anyou_id)
        print 'targetDMs : ', targetDMs
        anyou_type = anyou_type_assign
    else:
        seg_out = clean_str(src)
        srcfile = model_basedir + 'temp.txt'
        tmpfile = codecs.open(srcfile, 'w', 'utf-8')
        tmpfile.write(seg_out)
        tmpfile.close()
        # 判断是民事还是刑事
        if anyou_type == "":
            # model = model_basedir + 'mix/bin/mmix.bin'
            model = 'mmix'
            # cmdresult = callfastTextService(doc_hit, model, srcfile)
            # testresult = cmdresult.rstrip('\n').split(' ')
            cmdresult = callfastText(doc_hit, model, 'mix', srcfile)
            testresult = cmdresult[0].rstrip('\n').split(' ')
            if len(testresult) > 0:
                if testresult[0] == '__label__1':
                    anyou_type = 'minshi'
                else:
                    anyou_type = 'xingshi'

        #
        # 获取向量
        if anyou_type == 'xingshi':
            model_id = '1'
        else:
            model_id = '9000'

        # 调用fastText 预测案由一级分类
        # cmdresult = callfastTextService(doc_hit,anyou_type,srcfile)
        # testresult = cmdresult.rstrip('\n').split(' ')
        cmdresult = callfastText(doc_hit, model_id, anyou_type, srcfile)
        testresult = cmdresult[0].rstrip('\n').split(' ')

        # 获取分类结果
        result = {}
        result['status'] = 'ok'
        result['labels'] = []
        i = 0
        while i < len(testresult):
            label = {}
            label['prob'] = testresult[i + 1]
            if float(label['prob']) < 0.1:
                break
            if anyou_type == 'xingshi':
                label_map_key = xingshi_firstlist[0].get('DM') + testresult[i]
                label['DM'] = xingshi_label_map.get(label_map_key.rstrip('\n')).get('DM')
                label['anyou_label'] = xingshi_label_map.get(label_map_key.rstrip('\n')).get('MC')
                if float(label['prob']) > 0.1:
                    label['子类'] = recursiveclassify(label['DM'], anyou_type, xingshi_label_map, srcfile)
            else:
                label_map_key = minshi_firstlist[0].get('DM') + testresult[i]
                label['DM'] = minshi_label_map.get(label_map_key.rstrip('\n')).get('DM')
                label['anyou_label'] = minshi_label_map.get(label_map_key.rstrip('\n')).get('MC')
                if float(label['prob']) > 0.1:
                    label['子类'] = recursiveclassify(label['DM'], anyou_type, minshi_label_map, srcfile)
            # if i==0:
            result['labels'].append(label)
            i += 2
            if float(label['prob']) > 0.9:
                break

        # 根据分类确定候选文档集合，计算距离
        # 获取最底层子节点
        labelroot = result['labels']
        targetDMs = list()
        dict_target_DMs = dict()
        for label in labelroot:
            getLeaves(label, dict_target_DMs)

        targetDMs = dict_target_DMs.keys()

        sort_target_prob = sorted(dict_target_DMs.items(), key=lambda k: k[1], reverse=True)
        print 'targetDMs I: ', sort_target_prob
        avg_prob = sum(dict_target_DMs.values()) / len(dict_target_DMs)
        sim_prob = 0.5
        sim_prob = min(sim_prob, avg_prob)
        sim_target_num = 4
        sim_target_num = max(len(targetDMs) / 2, sim_target_num)
        sim_target = []

        for target_one in sort_target_prob:
            if len(sim_target) >= sim_target_num:
                break
            if target_one[1] >= sim_prob:
                sim_target.append(target_one[0])

        targetDMs = sim_target
        print 'targetDMs I: ', targetDMs

        if 'assign' in if_classify:
            para_anyou_id = re.compile(r'\d+').findall(anyou_id)
            for data_id in para_anyou_id:
                if data_id not in targetDMs:
                    targetDMs.append(data_id)
            print 'targetDMs II: ', targetDMs

    print 'targetDMs III: ', targetDMs

    """
    similar-text matrix vector
    """
    result_doc_list = []

    # mp_server
    dict_mp_server = dict()
    for i_server_url in mp_server_url:
        dict_mp_server[i_server_url] = list()

    mp_url_len = len(mp_server_url)
    for i in range(len(targetDMs)):
        dict_mp_server[mp_server_url[i % mp_url_len]].append(targetDMs[i])

    pool = multiprocessing.Pool(processes=len(mp_server_url))
    res_docs = list()

    for k, v in dict_mp_server.items():
        if v:
            simi_param = dict()
            simi_param['simisrc'] = str(src)
            simi_param['anyoutype'] = anyou_type
            simi_param['minsentlen'] = 6
            simi_param['maxsentlen'] = 12
            simi_param['anyouids'] = ','.join(v)
            res_docs.append(pool.apply_async(run_mp_server, (k, simi_param)))
    pool.close()
    pool.join()
    selected_docs = list()
    for res in res_docs:
        selected_docs.extend(res.get())

    selected_docs = [res_doc.split('\t') for res_doc in selected_docs]
    selected_docs.sort(key=lambda s: s[2], reverse=True)
    selected_docs = selected_docs[0:10]

    # sp_server
    # simi_server_url = 'http://172.23.4.87:10102/mpJavaSimilarAnyou'
    # simi_param = dict()
    # simi_param['simisrc'] = str(src)
    # simi_param['anyoutype'] = anyou_type
    # simi_param['minsentlen'] = 6
    # simi_param['maxsentlen'] = 12
    # simi_param['anyouids'] = ','.join(targetDMs)
    #
    # data_simi_param = urllib.urlencode(simi_param)
    # req = urllib2.Request(simi_server_url, data=data_simi_param)
    # response = urllib2.urlopen(req)
    # selected_docs = response.read()
    # selected_docs = str(selected_docs).split(';')
    # selected_docs = [res_doc.split('\t') for res_doc in selected_docs]

    for doc_val in selected_docs:
        # doc_val = doc.split('\t')
        if len(doc_val) < 2:
            continue
        newDoc = {}
        newDoc['DM'] = doc_val[0]
        newDoc['id'] = doc_val[1]
        if anyou_type == 'xingshi':
            newDoc['案由'] = xingshi_nodemap[doc_val[0]].get('MC')
        elif anyou_type == 'minshi':
            newDoc['案由'] = minshi_nodemap[doc_val[0]].get('MC')
        newDoc['相似度'] = doc_val[2]

        list_row_col = []
        target_list = [x for x in re.compile(r',').split(doc_val[3][1:len(doc_val[3]) - 1])]
        for i in range(len(target_list)):
            one_row_col = [x for x in re.compile(r' ').split(target_list[i])]
            if i == 0:
                list_row_col.append((int(one_row_col[0]), int(one_row_col[1])))
            else:
                list_row_col.append((int(one_row_col[1]), int(one_row_col[2])))

        newDoc['DATASYM'] = list_row_col

        result_doc_list.append(newDoc)

    for i in range(len(result_doc_list)):
        print  result_doc_list[i]['DM'], '---', result_doc_list[i]['id'], result_doc_list[i]['相似度']

    docloc = {}
    loc = 0
    docset = set()
    for doc in result_doc_list:
        docloc[doc['DM'] + '-' + str(doc['id'])] = loc
        loc += 1
        docset.add(doc['DM'])
    for targetDM in targetDMs:
        if targetDM not in docset:
            continue
        num = 0
        srcfile = model_basedir + anyou_type + '/src/' + targetDM + '.txt'
        fp = open(srcfile)
        line = fp.readline()
        gotten = 0
        while line:
            doc_index = targetDM + '-' + str(num)
            if doc_index in docloc.keys():
                result_doc_list[docloc[doc_index]]['src'] = line
                gotten += 1
                if gotten >= len(result_doc_list):
                    break
            num = num + 1
            line = fp.readline()
    print 'Elapse time ' + str((time.time() - start_time))
    return result_doc_list


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write(
            "<html><head><title>案件分析调用接口</title></head><body>案件分析调用接口  &nbsp;&nbsp;&nbsp;<a href=/demo>api调用demo</a>　 &nbsp;&nbsp;&nbsp;")
        self.write("<br><br>方法1：   anyouClassify ")
        self.write("<br>功能：   完成对案由的分类 ")
        self.write("<br>调用方法：http请求   请求URL：  http://$host:$port/anyouClassify")
        self.write("<br>请求类型：post    ")
        self.write("<br>请求参数：序列化后的json对象，包括２个字段，必选字段：src(原文)，可选字段：anyou_type(案由类型:minshi或者xingshi)")
        self.write("<br>编码支持：utf-8 ")
        self.write("<br>返回结果：Json对象序列化后的字符串 ")
        self.write("<br>")

        self.write("<br>方法2：  similarCase ")
        self.write("<br>功能：   根据一段文本匹配相似案例 ")
        self.write("<br>调用方法：http请求   请求URL：  http://$host:$port/similarCaseDemo")
        self.write("<br>请求类型：post    ")
        self.write("<br>请求参数：序列化后的json对象，包括２个字段，必选字段：src(原文)，可选字段：anyou_type(案由类型:minshi或者xingshi)")
        self.write("<br>编码支持：utf-8 ")
        self.write("<br>返回结果：Json对象序列化后的字符串")
        self.write("<br>")

        self.write("<br>方法３： zhengju ")
        self.write("<br>功能：   完成对证据名称的分类 ")
        self.write("<br>调用方法：http请求   请求URL：  http://$host:$port/zhengjuClassify")
        self.write("<br>请求类型：post    ")
        self.write("<br>请求参数：序列化后的json对象　包括１个字段：src(原文) ")
        self.write("<br>编码支持：utf-8 ")
        self.write("<br>返回结果：Json对象序列化后的字符串 ")
        self.write("</body></html>")


class SimilarCaseDemoHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('<html><body><form action="/similarCaseDemo" method="post">'
                   '<p><a href=/demo>案由分类demo</a> &nbsp;&nbsp;&nbsp;<b>相似案例查找demo　</b>&nbsp;&nbsp;&nbsp;<a href=/zhengju>证据分类demo</a> </p>'
                   '<p>案件基本情况</p>'
                   '<textarea name="message" style="width:600px;height:300px;">'
                   """原告秦皇岛日飞昕虹仪器仪表有限公司诉称，原、被告于2009年5月5日、2009年6月12日分别签订了《技术服务合同》，约定由原告为被告所承包的黄骅港工程提供海上施工作业中的测量、定位服务。合同签订后，原告按约定从2009年5月至12月期间为被告提供了相应技术服务，合计应收费45．3万元。但被告实际给付费用17．2万元，并于2011年期间给付5万元，尚欠23．1万元至今未付，原告就欠款事宜多次找被告催要，被告一直推拖至今。故原告诉至法院，要求判令被告给付服务费人民币23．1万元及利息（自2010年1月1日起至判决生效日止，按中国人民银行同期贷款利率计算）
                   """
                   '</textarea>'
                   '<p>&nbsp;</p>'
                   '<label><input name="if_classify" type="checkbox" value="identy" checked="checked"/>自动识别案由</label>'
                   '&nbsp;&nbsp;&nbsp;'
                   '<select name="anyou_type" style="width:100px;">'
                   '<option value="">自动识别</option>'
                   '<option value="minshi">民事</option>'
                   '<option value="xingshi">刑事</option>'
                   '</select>'
                   '<p>&nbsp;</p>'
                   '<label><input name="if_classify" type="checkbox" value="assign" />指定输入案由</label>'
                   '&nbsp;&nbsp;&nbsp;'
                   '<select name="correct_anyou_type" style="width:100px;">'
                   '<option value="minshi">民事</option>'
                   '<option value="xingshi">刑事</option>'
                   '</select>'
                   '&nbsp;&nbsp;&nbsp;'
                   '<input type="text" name="classify_id" />'
                   '(例: 9000，9001)'
                   '<br>'
                   '<a href=/anyou/xingshi_classify.json>刑事案由参考</a>&nbsp;&nbsp;&nbsp;'
                   '<a href=/anyou/minshi_classify.json>民事案由参考</a> </p>'
                   '<p>&nbsp;</p>'
                   '界面显示'
                   '&nbsp;&nbsp;&nbsp;'
                   '<label><input name="view_similar" type="radio" value="simi_line" checked="checked"/>短句匹配</label>'
                   '<label><input name="view_similar" type="radio" value="simi_word" />案例匹配</label>'
                   '<br><br> <input type="submit" value="Submit">'
                   '</form>'
                   '</body></html>')

    def post(self):
        self.set_header("Content-Type", "text/html")
        src = self.get_argument("message")
        if_classify = self.get_arguments("if_classify")
        if_classify = [str(x) for x in if_classify]
        view_similar = str(self.get_argument("view_similar"))
        anyou_id = self.get_argument("classify_id")
        anyou_type = self.get_argument("anyou_type")
        anyou_type_assign = self.get_argument("correct_anyou_type")
        similog.info(self.request.remote_ip + ', similar demo: ' + src)
        data = {}
        data["hits"] = 10
        data["src"] = src
        result = get_simi_docs(data["src"], data["hits"], anyou_type, str(anyou_type_assign), str(anyou_id),
                               if_classify)
        stopword_set = set(["年", "月", "日", "的", "了", "将", "诉称"
                               , "后", "于", "并", "但", "与", "元", "万元"
                               , "”", "、", "《", "》", "：", "；", "，", "。"])
        src_word_list = jieba.cut(src, cut_all=False)
        src_word_set = set()
        digits = re.compile(r"\d+")
        for word in src_word_list:
            digitmatch = re.match(digits, word)
            if digitmatch != None:
                continue
            if word.encode() not in stopword_set:
                src_word_set.add(word)

        from matrixText.matrix_seg import seg_sentence
        src_list = seg_sentence(str(src))

        self.write('<!DOCTYPE html>'
                   '<html><head>'
                   '<meta http-equiv="content-type" content="text/html;charset=utf-8">'
                   '<title>相似案例结果</title>')
        self.write("</head>")
        self.write("<body>")
        self.write("<div>")

        self.write('<table border = "1">'
                   '<tr>'
                   '<th style="width: 40%">' + '源文' + '</th>'
                                                      '<th style="width: 60%">相似案例</th>'
                                                      '</tr>'
                   )

        separation = '<br>'
        i = 1
        for res_one in result:
            if "src" not in res_one.keys():
                continue

            if 'simi_line' == view_similar:

                self.write('<tr>')
                self.write('<td>')
                self.write("<p>" + str(i) + ":&nbsp;&nbsp;&nbsp;&nbsp;")
                self.write(separation)

                res_test_sym = list()
                res_norm_sym = list()
                for j in range(len(res_one['DATASYM'])):
                    res_test_sym.append(res_one['DATASYM'][j][0])
                    res_norm_sym.append(res_one['DATASYM'][j][1])

                for j in range(len(src_list)):
                    if j in res_test_sym:
                        if res_test_sym.index(j) == 0:
                            self.write("<b><font color=\"red\">" + '[ ' + str(j) + ' - ' + str(res_norm_sym[0])
                                       + ' ]' + '<sub>' + str(0) + '</sub>' + ': ' + src_list[j] + "</font></b>")
                            self.write(separation)
                        elif res_test_sym.index(j) in [1, 2, 3]:
                            self.write("<b><font color=\"blue\">" + '[ ' + str(j) + ' - ' + str(
                                res_norm_sym[res_test_sym.index(j)])
                                       + ' ]' + '<sub>' + str(res_test_sym.index(j)) + '</sub>' + ': '
                                       + src_list[j] + "</font></b>")
                            self.write(separation)
                        else:
                            self.write("<b><font color=\"olive\">" + '[ ' + str(j) + ' - ' + str(
                                res_norm_sym[res_test_sym.index(j)])
                                       + ' ]' + '<sub>' + str(res_test_sym.index(j)) + '</sub>' + ': ' + "</font></b>")
                            self.write(src_list[j])
                            self.write(separation)
                    elif src_list[j] and not re.compile(r'^\s*\n*$').match(src_list[j]):
                        self.write('[' + str(j) + ']: ' + src_list[j])
                        self.write(separation)
                self.write('</td>')

                self.write('<td>')
                self.write("<p>" + str(i) + ":&nbsp;&nbsp;&nbsp;&nbsp;")
                self.write(separation)
                doc_list = seg_sentence(res_one["src"])

                for j in range(len(doc_list)):
                    if j in res_norm_sym:
                        if res_norm_sym.index(j) == 0:
                            self.write("<b><font color=\"red\">" + '[ ' + str(j) + ' - ' + str(res_test_sym[0])
                                       + ' ]' + '<sub>' + str(0) + '</sub>' + ': ' + doc_list[j] + "</font></b>")
                            self.write(separation)
                        elif res_norm_sym.index(j) in [1, 2, 3]:
                            self.write(
                                "<b><font color=\"blue\">" + '[ ' + str(j) + ' - ' + str(
                                    res_test_sym[res_norm_sym.index(j)])
                                + ' ]' + '<sub>' + str(res_norm_sym.index(j)) + '</sub>' + ': '
                                + doc_list[j] + "</font></b>")
                            self.write(separation)
                        else:
                            self.write(
                                "<b><font color=\"olive\">" + '[ ' + str(j) + ' - ' + str(
                                    res_test_sym[res_norm_sym.index(j)])
                                + ' ]' + '<sub>' + str(res_norm_sym.index(j)) + '</sub>' + ': ' + "</font></b>")
                            self.write(doc_list[j])
                            self.write(separation)
                    elif doc_list[j] and not re.compile(r'^\s*\n*$').match(doc_list[j]):
                        self.write('[' + str(j) + ']: ' + doc_list[j])
                        self.write(separation)

                self.write('</td>')
                self.write('</tr>')

            elif 'simi_word' == view_similar or view_similar == []:

                self.write('<tr>')
                self.write('<td>')
                self.write("<p>" + str(i) + ":&nbsp;&nbsp;&nbsp;&nbsp;")
                self.write(separation)

                self.write(str(src))
                self.write('</td>')

                self.write('<td>')
                self.write("<p>" + str(i) + ":&nbsp;&nbsp;&nbsp;&nbsp;")
                self.write(separation)

                target_word_list = jieba.cut(res_one["src"], cut_all=False)
                sameCount = 0
                sameWords = ""
                for word in target_word_list:
                    if word in src_word_set:
                        sameWords += word
                        sameCount += 1
                    else:
                        if sameCount >= 3:
                            self.write("<b><font color=\"green\">" + sameWords + "</font></b>")
                        else:
                            self.write(sameWords)
                        sameCount = 0
                        sameWords = ""
                        self.write(word)
                if sameCount > 0:
                    if sameCount >= 2:
                        self.write("<b><font color=\"green\">" + sameWords + "</font></b>")
                    else:
                        self.write(sameWords)

                self.write('</td>')
                self.write('</tr>')

            i += 1

        self.write('</table>')
        self.write("</div>")
        self.write("</body></html>")


settings = {
    'static_path': os.path.join(os.path.dirname(__file__), 'anyou'),
    'cookie_secret': '61oETzKXQAGaYdkL5gEmGeJJFuYh7EQnp2XdTP1o/Vo=',
    'login_url': '/login',
    'xsrf_cookies': False,
}

application = tornado.web.Application([
    (r'/', MainHandler),
    (r'/similarCaseDemo', SimilarCaseDemoHandler),
    (r'/anyou/(xingshi_classify\.json)', tornado.web.StaticFileHandler, dict(path=settings['static_path'])),
    (r'/anyou/(minshi_classify\.json)', tornado.web.StaticFileHandler, dict(path=settings['static_path'])),
], **settings)

if __name__ == '__main__':
    cp = ConfigParser.SafeConfigParser()
    cp.read('calc_simi_anyou.conf')
    work_dir = cp.get('server', 'work_dir')
    server_port = cp.get('server', 'port')

    mp_server_ip_str = cp.get('java_mp', 'server_url')
    mp_server_url = [s.strip() + 'mpJavaSimilarAnyou' for s in mp_server_ip_str[1:-1].split(',')]

    node.loadConfig(work_dir + 'anyou/AY_minshi.xml', minshi_firstlist, minshi_nodemap, minshi_label_map)
    node.loadConfig(work_dir + 'anyou/AY_xingshi.xml', xingshi_firstlist, xingshi_nodemap, xingshi_label_map)

    similog = FinalLogger.getLogger()
    similog.info('anyou classify')

    print '---start server---'
    similog.info('---start server---')

    application.listen(server_port)
    tornado.ioloop.IOLoop.instance().start()
