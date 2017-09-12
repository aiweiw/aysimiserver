# coding=utf-8

from gensim import corpora, models, similarities
import jieba
import jieba.posseg as pseg
import time

class MyCorpus(object):
    def __iter__(self):
        for line in open('mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())


documents = ["Shipment of gold damaged in a fire",
             "Delivery of silver arrived in a silver truck",
             "Shipment of gold arrived in a truck"]

documents_ch = ['原告秦皇岛日飞昕虹仪器仪表有限公司诉称，原、被告于2009年5月5日、2009年6月12日分别签订了《技术服务合同》，'
             '约定由原告为被告所承包的黄骅港工程提供海上施工作业中的测量、定位服务。合同签订后，'
             '原告按约定从2009年5月至12月期间为被告提供了相应技术服务，合计应收费45．3万元。'
             '但被告实际给付费用17．2万元，并于2011年期间给付5万元，尚欠23．1万元至今未付，'
             '原告就欠款事宜多次找被告催要，被告一直推拖至今。故原告诉至法院，'
             '要求判令被告给付服务费人民币23．1万元及利息（自2010年1月1日起至判决生效日止，按中国人民银行同期贷款利率计算）。',
             '原告上海舜驰投资管理有限公司诉称：原、被告于2013年5月9日签订山阳镇龙皓路298号商业项目前期商业定位策划服务合同一份，'
             '约定原告的服务内容为市场调研、商业定位策划，原告为被告商业布局制作详尽的商业规划方案以及为被告提供项目后续招商战略的'
             '一系列可行性和基础性的准备及支持服务；合同时限为2013年5月10日至6月9日，共计30天；'
             '商业策划总计费用为人民币（以下币种同）10万元，分两次支付，'
             '第一次费用自合同签订之日被告提供详细项目资料起2日内原告向被告提供相应金额的发票，被告应在收到原告提供的发票后3日内支付6万元作为首期费用，'
             '第二次费用在商业策划项目结束，被告以当面或邮件以及书面形式收到并认可原告交付的商业策划报告和发票后的5个工作日内，'
             '被告支付4万元；如被告无正当理由逾期支付原告费用的，被告除继续履行支付义务外，还应当按照应付费用的日万分之一支付滞纳金。'
             '合同签订后，原告于2013年5月14日向被告开具了一份金额为6万元的发票，被告于2013年5月27日向原告支付了6万元。'
             '原告于2013年6月8日以邮件形式向被告发送了“金山区山阳镇龙皓”文件完成了服务。同日原告还向被告开具了一份4万元的发票，'
             '被告于2013年12月9日向原告支付了1万元，尚欠3万元至今未付。故原告诉至法院，请求判令被告支付服务费3万元及逾期利息2000元。'
             '庭审中原告明确利息诉请为要求被告支付自2013年6月9日至2014年10月1日，按照每日万分之一计算的滞纳金。',
             '原告上海晋晓实业有限公司诉称：原、被告签订《消防设施维护保养技术服务合同》，约定由原告对被告的消防设施进行维护保养，'
             '期限为2010年7月1日至2012年6月31日，被告每季度支付17，786．25元。合同签订后，原告积极履行合同义务，'
             '至2012年6月31日维保服务合同到期，双方没有另行签订书面续约合同，但原告仍继续为被告提供维保服务至2012年12月31日。'
             '原、被告因先前一起消防设施施工合同善后事宜协商未果产生纠纷，被告将维保合同中的维保费用一并予以扣留。经多次催讨未果，'
             '故原告诉至法院，请求判令：1、被告偿付原告服务费71，145元；2、被告支付原告逾期付款的利息损失（以71，145元为基数，'
             '自2013年1月2日起算至实际清偿之日止，按照中国人民银行同期贷款利率计算）。',
             '原告怡生乐居公司诉称：原告与被告有长期的业务合作关系，原告为被告提供网络广告发布服务，被告向原告支付服务费用。'
             '2013年1月，原告与被告签订了《网络广告发布合同》，合同约定：被告在2013年2月1日至2014年1月31日期间在原告处投放价值'
             '50万元人民币的网络广告，并以《广告发布排期表》形式约定了投放时间、广告类型等。合同签订后，原告依照合同约定履行了发'
             '布广告的义务，但被告支付了5万元款项后一直未向原告支付余款，至今尚欠原告广告服务费377084元。经原告多次催要，被告一'
             '直未予偿还。现原告诉至法院，请求判令：1、被告向原告支付广告发布服务费377084元；2、被告支付原告2013年11月15日'
             '至2014年7月1日的迟延支付违约金78056元。诉讼费由被告承担。 被告柯拉尼家居公司经本院送达起诉状副本及开庭传票，'
             '未到庭应诉，亦未提交答辩意见。',
             '原告诉称，2014年6月，原告法定代理人在网上发现刘洋以刘佳名义在面瘫保愈之家群里发布面瘫包治信息后与被告联系，'
             '2014年7月7日与被告委托代理人刘佳签订了协议书，该协议约定治疗期限为3个月，治疗目标为恢复到正常面容，'
             '如未完全达到治疗目的，免费延期治疗三个月，若仍未痊愈，把医疗费全额退回。协议签订后，原告足额交付面瘫康复款12000元，'
             '并按期接受治疗。治疗期限届满后，原告没有完全恢复正常面容，尽管按照被告要求增加了三个月的治疗，面容仍不见好转。'
             '原告要求被告退款，被告推脱至今，故提起诉讼，要求被告返还面瘫康复款12000元及利息（自2014年7月7日起至给付之日止，'
             '按照中国人民银行同期同类贷款利率计算），并赔偿误工费10586．35元、交通费3608元、住宿费5240元、'
             '代理费2500元及精神损害赔偿等各项经济损失共计5万元，本案诉讼费由被告负担。 ']

texts = [[word for word in document.lower().split()] for document in documents]

texts = []

dir_file = 'dataTfidf/9813.txt'
f = open(dir_file, 'r')
line = f.readline()
line_num = 1
while line:
    seg_line = []
    seg_list = pseg.cut(line)
    for word, flag in seg_list:
        seg_line.append(word)
    texts.append(seg_line)
    line = f.readline()
    print '-----line number:', line_num
    line_num += 1
f.close()
time.sleep(0)


# for line in documents_ch:
#     seg_line = []
#     seg_list = pseg.cut(line)
#     for word, flag in seg_list:
#         seg_line.append(word)
#     texts.append(seg_line)


print texts

print 'token2id:'
dictionary = corpora.Dictionary(texts)
print dictionary
print dictionary.token2id

print 'word-freq corpus, doc2bow:'
corpus = [dictionary.doc2bow(text) for text in texts]
print corpus

print 'freq-idf:'
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

for doc in corpus_tfidf:
    print doc

print 'dfs-idfs:'
print tfidf.dfs
print tfidf.idfs

# lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
# print 'show topics model:'
# lsi.print_topics(3)
#
# print 'doc-lsi projection:'
# corpus_lsi = lsi[corpus_tfidf]
# for doc in corpus_lsi:
#     print doc
#
# index = similarities.MatrixSimilarity(lsi[corpus])
# query = "gold silver truck"
# query_ch = '原告秦皇岛日飞昕虹仪器仪表有限公司诉称，原、被告于2009年5月5日、2009年6月12日分别签订了《技术服务合同》，' \
#            '约定由原告为被告所承包的黄骅港工程提供海上施工作业中的测量、定位服务。合同签订后，' \
#            '原告按约定从2009年5月至12月期间为被告提供了相应技术服务，合计应收费45．3万元。但被告实际给付费用17．2万元，' \
#            '并于2011年期间给付5万元，尚欠23．1万元至今未付，原告就欠款事宜多次找被告催要，被告一直推拖至今。故原告诉至法院，' \
#            '要求判令被告给付服务费人民币23．1万元及利息（自2010年1月1日起至判决生效日止，按中国人民银行同期贷款利率计算）。'
#
# seg_query = pseg.cut(query_ch)
# list_query = []
# for word, flag in seg_query:
#     list_query.append(word)
#
# print 'query-lsi projection:'
# query_bow = dictionary.doc2bow(list_query)
# query_lsi = lsi[query_bow]
# print query_lsi
#
# print 'cos_dist:'
# sims = index[query_lsi]
# print list(enumerate(sims))
#
# sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
# print sort_sims



lda = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=500)
print 'show topics model:'
lda.print_topics(3)

print 'doc-lsi projection:'
corpus_lsi = lda[corpus_tfidf]
for doc in corpus_lsi:
    print doc

index = similarities.MatrixSimilarity(lda[corpus])
query = "gold silver truck"
query_ch = '原告秦皇岛日飞昕虹仪器仪表有限公司诉称，原、被告于2009年5月5日、2009年6月12日分别签订了《技术服务合同》，' \
           '约定由原告为被告所承包的黄骅港工程提供海上施工作业中的测量、定位服务。合同签订后，' \
           '原告按约定从2009年5月至12月期间为被告提供了相应技术服务，合计应收费45．3万元。但被告实际给付费用17．2万元，' \
           '并于2011年期间给付5万元，尚欠23．1万元至今未付，原告就欠款事宜多次找被告催要，被告一直推拖至今。故原告诉至法院，' \
           '要求判令被告给付服务费人民币23．1万元及利息（自2010年1月1日起至判决生效日止，按中国人民银行同期贷款利率计算）。'

seg_query = pseg.cut(query_ch)
list_query = []
for word, flag in seg_query:
    list_query.append(word)

print 'query-lsi projection:'
query_bow = dictionary.doc2bow(list_query)
query_lsi = lda[query_bow]
print query_lsi

print 'cos_dist:'
sims = index[query_lsi]
print list(enumerate(sims))

sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
print sort_sims