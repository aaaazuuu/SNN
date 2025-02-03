import os
import numpy as np

import got_studied_w_path_list.brass_piano_clarinet_guiter_xylophone_al_random_path

w_a1 = got_studied_w_path_list.brass_piano_clarinet_guiter_xylophone_al_random_path.w_a1
w_a2 = got_studied_w_path_list.brass_piano_clarinet_guiter_xylophone_al_random_path.w_a2
w_a3 = got_studied_w_path_list.brass_piano_clarinet_guiter_xylophone_al_random_path.w_a3


# cos類似度計算
def CosSim_w_a1():
    a = w_a1.copy().T
    b = w_a1.copy().T

    a_dis = np.sqrt((a ** 2).sum(axis=1))  # ユークリッド距離　l2norm →　割ると正規化される 過学習を防ぐためにも使えるから、勉強しておくと良い
    b_dis = np.sqrt((b ** 2).sum(axis=1))

    a = (a.T / (a_dis + 1e-8)).T  # ユークリッド距離　l2norm →　割ると正規化される
    b = (b.T / (b_dis + 1e-8)).T  # ユークリッド距離　l2norm →　割ると正規化される

    cossim_1 = np.matmul(a, b.T)
    # print(cossim_1.shape)

    t1 = np.triu(cossim_1, k=1)

    # 対角線を除いた平均類似度の計算
    s1 = cossim_1[np.triu_indices_from(cossim_1, k=1)]
    mean_cossim_1 = np.mean(s1)

    return cossim_1, mean_cossim_1, s1, t1


def CosSim_w_a2():
    a = w_a2.copy().T
    b = w_a2.copy().T

    a_dis = np.sqrt((a ** 2).sum(axis=1))  # ユークリッド距離　l2norm →　割ると正規化される 過学習を防ぐためにも使えるから、勉強しておくと良い
    b_dis = np.sqrt((b ** 2).sum(axis=1))

    a = (a.T / (a_dis + 1e-8)).T  # ユークリッド距離　l2norm →　割ると正規化される
    b = (b.T / (b_dis + 1e-8)).T  # ユークリッド距離　l2norm →　割ると正規化される

    cossim_2 = np.matmul(a, b.T)
    # print(cossim_2.shape)

    # 対角線を除いた平均類似度の計算
    s2 = cossim_2[np.triu_indices_from(cossim_2, k=1)]
    mean_cossim_2 = np.mean(s2)

    return cossim_2, mean_cossim_2, s2


def CosSim_w_a3():
    a = w_a2.copy().T
    b = w_a2.copy().T

    a_dis = np.sqrt((a ** 2).sum(axis=1))  # ユークリッド距離　l2norm →　割ると正規化される 過学習を防ぐためにも使えるから、勉強しておくと良い
    b_dis = np.sqrt((b ** 2).sum(axis=1))

    a = (a.T / (a_dis + 1e-8)).T  # ユークリッド距離　l2norm →　割ると正規化される
    b = (b.T / (b_dis + 1e-8)).T  # ユークリッド距離　l2norm →　割ると正規化される

    cossim_3 = np.matmul(a, b.T)
    # print(cossim_2.shape)
    # 対角線を除いた平均類似度の計算
    s3 = cossim_3[np.triu_indices_from(cossim_3, k=1)]
    mean_cossim_3 = np.mean(s3)

    return cossim_3, mean_cossim_3, s3