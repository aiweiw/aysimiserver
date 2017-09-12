# coding: utf-8

from operator import itemgetter

"""
Compute matrix-similarity
input: two matrix to computed
output: every matrix's N-most-similar-vectors
"""


# corr-distance-function
def corr_distance(x, y):
    """
    :param x: test data
    :param y: norm data
    :return: corr-distance
    """
    if len(x) != len(y):
        return

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


# cos-distance-function
def cos_distance(x, y):
    """
    :param x: test data
    :param y: norm data
    :return: corr-distance
    """
    if len(x) != len(y):
        return

    res_fenzi = map(lambda (a, b): a * b, zip(x, y))
    res_fenmu0 = map(lambda a: a ** 2, x)
    res_fenmu1 = map(lambda b: b ** 2, y)

    if sum(res_fenmu0) < 0.0000001 or sum(res_fenmu1) <= 0.000001:
        return 0.0

    return sum(res_fenzi) / (sum(res_fenmu0) * sum(res_fenmu1)) ** 0.5


# flatten list
def flatten_list(nested):
    """
    :param nested: nested list
    :return: flattern list
    """
    if isinstance(nested, list):
        for sub_list in nested:
            for item in flatten_list(sub_list):
                yield item
    else:
        yield nested


# distance-matrix
def dist_matrix(test_data, norm_data, weight_line=None, min_len_norm_data=5, len_link=None):
    """
    matrix-0: test_data's seq
    matrix-1: norm_data's seq
    matrix-2: multiply
    """
    if len(norm_data) < min_len_norm_data:
        return None

    if not weight_line:
        weight_line = [1.0] * len(norm_data)
    elif len(norm_data) != len(weight_line):
        return None

    len_deal = min(len(test_data), len(norm_data))
    if len_link:
        len_deal = min(len_deal, len_link)

    list_mul = []
    for i in range(len(test_data)):
        for j in range(len(norm_data)):
            temp_dist = corr_distance(test_data[i], norm_data[j])
            list_mul.append((i, j, temp_dist, weight_line[j] * temp_dist))
            # list_mul.append((i, j, weight_line[j] * abs(cos_distance(test_data[i], norm_data[j]))))

    #
    list_mul.sort(key=itemgetter(3), reverse=True)
    # list_mul.sort(key=itemgetter(2))

    list_result = []
    list_col = []
    list_row = []
    for list_one in list_mul:
        if len(list_result) >= len_deal:
            break
        if list_one[0] not in list_row and list_one[1] not in list_col:
            list_row.append(list_one[0])
            list_col.append(list_one[1])
            list_result.append(list_one)

    return list_result

    # test_res = []
    # norm_res = []
    # for list_one in list_result:
    #     test_res.append(test_data[list_one[0]])
    #     norm_res.append(norm_data[list_one[1]])

    # flatten list, compute distance, ...
    # return test_res, norm_res
