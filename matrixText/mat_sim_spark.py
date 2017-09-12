# encoding: utf-8

from operator import itemgetter

import time
from pyspark import SparkConf, SparkContext
from pyspark.storagelevel import StorageLevel

import sys

# vec-similar-data
# file_anyou = '../anyou/minshi/cbowVectAnsj/9178.txt.out.vect'
# file_anli = '../anyou/similar.test'
# file_anli = '../anyou/temp.test'

# matrix-similar-data
file_anyou = '../anyou/minshi/cbowVectAnsj/9130.txt.out.ansj.vect'
file_anli = '../anyou/simimat.test'

# file_anyou = 'hdfs://172.16.124.6:9000/test/matrixay/9130.txt.out.ansj.vect'
# file_anli = 'hdfs://172.16.124.6:9000/test/matrixay/simimat.test'

# file_anyou = 'hdfs://172.16.124.6:9000/test/matrixay/9131.txt.out.vect'
# file_anli = 'hdfs://172.16.124.6:9000/test/matrixay/similar.test'

index_dist = dict()
index_dist_max10 = dict()
reduce_dist_max10 = dict()
dist_max_num = 10


def corr_dist(x, y):
    """
    :param x: test data
    :param y: norm data
    :return: corr-distance
    """
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


def corr_distance(x, y, index):
    """
    :param x: test data
    :param y: norm data
    :return: corr-distance
    """
    x_split = x.split()
    y_split = y.split()

    x = [float(param) for param in x_split]
    y = [float(param) for param in y_split]

    dist = corr_dist(x, y)
    # index_dist_max10[index] = dist
    # while len(index_dist_max10) > dist_max_num:
    #     dist_min = min(index_dist_max10.values())
    #     index_dist_max10.pop([k for k, v in index_dist_max10.items() if v == dist_min][0])
    # if index_dist_max10:
    #     dist_min = min(index_dist_max10.values())
    #     if dist > dist_min:
    #         index_dist_max10.pop([k for k, v in index_dist_max10.items() if v == dist_min][0])
    #         index_dist_max10[index] = dist
    #     while len(index_dist_max10) > dist_max_num:
    #         dist_min = min(index_dist_max10.values())
    #         index_dist_max10.pop([k for k, v in index_dist_max10.items() if v == dist_min][0])
    # else:
    #     index_dist_max10[index] = dist

    # return dist, index_dist_max10
    return dist

    # if len(x) != len(y):
    #     return 0.0
    #
    # # avg = (sum(x) + sum(y)) / (len(x) + len(y))
    # x_avg = sum(x) / len(x)
    # y_avg = sum(y) / len(y)
    # x_tran = map(lambda a: a - x_avg, x)
    # y_tran = map(lambda b: b - y_avg, y)
    #
    # res_fenzi = map(lambda (a, b): a * b, zip(x_tran, y_tran))
    # res_fenmu0 = map(lambda a: a ** 2, x_tran)
    # res_fenmu1 = map(lambda b: b ** 2, y_tran)
    #
    # if sum(res_fenmu0) < 0.0000001 or sum(res_fenmu1) <= 0.000001:
    #     return 0.0
    #
    # return sum(res_fenzi) / (sum(res_fenmu0) * sum(res_fenmu1)) ** 0.5


def corr_dist_parti(iterator):
    for val in iterator:
        # print val[0], val[1], val[2]
        # print '------'
        # x_split = val[1].split()
        # y_split = val[2].split()
        #
        # x = [float(param) for param in x_split]
        # y = [float(param) for param in y_split]

        dist = corr_dist(val[1], val[2])
        index_dist_max10[val[0]] = dist
        while len(index_dist_max10) > dist_max_num:
            dist_min = min(index_dist_max10.values())
            index_dist_max10.pop([k for k, v in index_dist_max10.items() if v == dist_min][0])
            # if index_dist_max10:
            #     dist_min = min(index_dist_max10.values())
            #     if dist > dist_min:
            #         index_dist_max10.pop([k for k, v in index_dist_max10.items() if v == dist_min][0])
            #         index_dist_max10[val[0]] = dist
            #     while len(index_dist_max10) > dist_max_num:
            #         dist_min = min(index_dist_max10.values())
            #         index_dist_max10.pop([k for k, v in index_dist_max10.items() if v == dist_min][0])
            # else:
            #     index_dist_max10[val[0]] = dist

    print '-----'
    yield index_dist_max10


def corr_dist_matrix(x_mat, y_mat, index, vec_dim=100):
    # getVector(String src, int dimLen, String textTfidf, ArrayList < Double > arrTextTfidf)
    x_split = x_mat.split()
    y_split = y_mat.split()

    x_matrix = [float(param) for param in x_split]
    y_matrix = [float(param) for param in y_split]

    if len(x_matrix) % vec_dim != 0 or len(y_matrix) % vec_dim != 0:
        return []

    xi_dim = len(x_matrix) / vec_dim
    yi_dim = len(y_matrix) / vec_dim

    x_matrix = [x_matrix[i * vec_dim: (i + 1) * vec_dim] for i in range(xi_dim)]
    y_matrix = [y_matrix[i * vec_dim: (i + 1) * vec_dim] for i in range(yi_dim)]

    sim_senten_len = min(len(x_matrix), len(y_matrix))

    list_mul = []
    for i in range(len(x_matrix)):
        for j in range(len(y_matrix)):
            temp_dist = corr_dist(x_matrix[i], y_matrix[j])
            list_mul.append((i, j, temp_dist))

    #
    list_mul.sort(key=itemgetter(2), reverse=True)

    list_result = []
    list_col = []
    list_row = []
    mean_dist = 0.0
    for list_one in list_mul:
        if len(list_result) >= sim_senten_len:
            break
        if list_one[0] not in list_row and list_one[1] not in list_col:
            list_row.append(list_one[0])
            list_col.append(list_one[1])
            list_result.append(list_one)
            mean_dist += list_one[2]

    mean_dist /= len(list_result)

    # index_dist_max10[index] = mean_dist
    # while len(index_dist_max10) > dist_max_num:
    #     dist_min = min(index_dist_max10.values())
    #     index_dist_max10.pop([k for k, v in index_dist_max10.items() if v == dist_min][0])

    # print '-----'
    # return list_result, mean_dist
    return mean_dist
    # return list_result, index_dist_max10


def corr_dist_partitions(iterator):
    for val in iterator:
        x_split = val[1].split()
        y_split = val[2].split()

        x_matrix = [float(param) for param in x_split]
        y_matrix = [float(param) for param in y_split]

        # x_matrix = val[1]
        # y_matrix = val[2]

        vec_dim = int(val[3])

        if len(x_matrix) % vec_dim != 0 or len(y_matrix) % vec_dim != 0:
            continue

        xi_dim = len(x_matrix) / vec_dim
        yi_dim = len(y_matrix) / vec_dim

        x_matrix = [x_matrix[i * vec_dim: (i + 1) * vec_dim] for i in range(xi_dim)]
        y_matrix = [y_matrix[i * vec_dim: (i + 1) * vec_dim] for i in range(yi_dim)]

        sim_senten_len = min(len(x_matrix), len(y_matrix))

        list_mul = []
        for i in range(len(x_matrix)):
            for j in range(len(y_matrix)):
                temp_dist = corr_dist(x_matrix[i], y_matrix[j])
                list_mul.append((i, j, temp_dist))

        list_mul.sort(key=itemgetter(2), reverse=True)

        list_result = []
        list_col = []
        list_row = []
        mean_dist = 0.0
        for list_one in list_mul:
            if len(list_result) >= sim_senten_len:
                break
            if list_one[0] not in list_row and list_one[1] not in list_col:
                list_row.append(list_one[0])
                list_col.append(list_one[1])
                list_result.append(list_one)
                mean_dist += list_one[2]

        mean_dist /= len(list_result)

        if not index_dist_max10 or len(index_dist_max10) < dist_max_num:
            index_dist_max10[val[0]] = mean_dist
            continue
        if mean_dist > min(index_dist_max10.values()):
            index_dist_max10[val[0]] = mean_dist

        while len(index_dist_max10) > dist_max_num:
            # print '------', len(index_dist_max10)
            dist_min = min(index_dist_max10.values())
            index_dist_max10.pop([k for k, v in index_dist_max10.items() if v == dist_min][0])

    print '-----'
    yield index_dist_max10


def reduce_partitions(*red_dist):
    """
    :param red_dist: reduce param
    :return:
    """
    # print ''
    # print '---new devide---'
    # print ''
    for dict_dist in red_dist:
        if not dict_dist:
            continue
        # print '--data', dict_dist
        for k, v in dict_dist.items():
            if not reduce_dist_max10 or len(reduce_dist_max10) < dist_max_num:
                reduce_dist_max10[k] = v
                continue
            if v > min(reduce_dist_max10.values()):
                reduce_dist_max10[k] = v
                # print '++++', k, v
            while len(reduce_dist_max10) > dist_max_num:
                dist_min = min(reduce_dist_max10.values())
                reduce_dist_max10.pop([k for k, v in reduce_dist_max10.items() if v == dist_min][0])

                # print 'reduce_dist', reduce_dist_max10


def parse(row):
    """
    :param row:
    :return:
    """
    list_num = list()
    for num in row:
        list_num.append(float(num))

    return list_num


def split(line):
    """
    :param line:
    :return:
    """
    return line.split()


if __name__ == '__main__':
    print 'begin'
    start_time = time.time()

    print sys.argv

    # v2.0.2
    # conf = SparkConf().setMaster('spark://172.16.124.5:7077').setAppName('simiComput')\
    #     .set('spark.executor.memory', '8G').set('spark.num.executors', '100').set('spark.default.parallelism', '1000')\
    #     .set('spark.driver.memory', '2G').set('spark.reducer.maxSizeInFlight', '96m')
    conf = SparkConf().setMaster('local').setAppName('simiComput') \
        .set('spark.executor.memory', '6g').set('spark.num.executors', '100').set('spark.default.parallelism', '1000')
    sc = SparkContext(conf=conf)
    matrix_anyou = sc.textFile(sys.argv[1]).cache()  # persist(StorageLevel.MEMORY_AND_DISK_SER)
    print '---NumPartition---'
    print matrix_anyou.getNumPartitions()
    print '---NumPartition---'

    simi = sc.textFile(sys.argv[2])
    simi_vec = simi.map(split).map(parse).collect()[0]
    simi_line = simi.collect()[0]

    # ---matrix similar distance---
    # print matrix_anyou.zipWithIndex().map(lambda anli: (anli[1], corr_dist_matrix(anli[0], simi_line))).sortBy(
    #     lambda s: s[1][1]).first()

    # print matrix_anyou.zipWithIndex().map(
    #     lambda anli: (corr_dist_matrix(anli[0], simi_line, anli[1]), anli[1])).top(10)

    # print heapq.nlargest(10, matrix_anyou.zipWithIndex().map(
    #     lambda anli: (anli[1], corr_dist_matrix(anli[0], simi_line))).collect(), key=lambda s: s[1][1])

    # print matrix_anyou.zipWithIndex().map(
    #     lambda anli: (anli[1], corr_dist_matrix(anli[0], simi_line, anli[1]))).lookup(key=lambda s: s[0] == 5000)

    # print matrix_anyou.zipWithIndex().map(
    #     lambda anli: (anli[1], corr_dist_matrix(anli[0], simi_line))).sortBy(lambda s: s[1]).first()

    print matrix_anyou.zipWithIndex().map(
        lambda param: (param[1], param[0], simi_line, 100)).mapPartitions(corr_dist_partitions).reduce(
        reduce_partitions)

    # ---vector similar distance---
    # print matrix_anyou.zipWithIndex().map(
    #     lambda anli: (corr_distance(anli[0], simi_line, anli[1]), anli[1])).top(10)

    # print heapq.nlargest(10, matrix_anyou.zipWithIndex().map(
    #     lambda anli: (anli[1], corr_distance(anli[0], simi_line))).collect(), key=lambda s: s[1])

    # print sorted(matrix_anyou.zipWithIndex().map(lambda anli: (anli[1], corr_distance(anli[0], simi_line))).collect(),
    #              key=lambda s: s[1][1], reverse=True)[0:10]

    # print matrix_anyou.zipWithIndex().mapPartitions(
    #     lambda anli: (anli[1], corr_distance(anli[0], simi_line, anli[1]))).cache().take(20)

    # print matrix_anyou.zipWithIndex().map(lambda param: (param[1], param[0], simi_line)).collect()

    # print matrix_anyou.map(split).map(parse).zipWithIndex().map(
    #     lambda param: (param[1], param[0], simi_vec)).mapPartitions(corr_dist_parti).reduce(reduce_partitions)

    print '......'
    print 'Final dist', reduce_dist_max10

    print 'time: ', time.time() - start_time

print 'ok'
