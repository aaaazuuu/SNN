import os


import numpy as np

from collections import Counter

import matplotlib.pyplot as plt


log_folder = os.path.abspath("../../log")


def main(log_time: str):
    # pathの作成
    log_folder_ = os.path.join(log_folder, log_time)
    npy_file_path = os.path.join(log_folder_, "w_a1.npy")

    # w_a1
    w_a1: np.ndarray = np.load(npy_file_path)

    # 画像の作成
    # fig = plt.figure() # 描写可能な領域を作る
    # axis = fig.add_subplot(1, 1, 1) # 描写可能な領域にグラフをプロットする領域を作る
    # axis.plot( # プロットの領域にグラフをプロットする
    #     np.arange(0, w_a2.shape[0]), # 0, 1, 2, ... , w_a2の行列数　の配列を作る
    #     # np.sum(w_a2, axis=1) # 軸１（列方向）に沿って和を出す
    # )
    # fig.savefig(os.path.join(log_folder_, "w_a2.svg")) # figを画像として保存する

    # パターンの数を取得
    def column_patterrns():
        column_patterns = [tuple(w_a1[:, i]) for i in range(w_a1.shape[1])]

        # 各パターンの出現回数をカウント
        pattern_counts = Counter(column_patterns)
        frequencies = list(pattern_counts.values())

        return frequencies
    fre = column_patterrns()

    column_patterns = [tuple(w_a1[:, i]) for i in range(w_a1.shape[1])]
    pattern_counts = Counter(column_patterns)
    print(pattern_counts)


    print(fre)
    print(np.size(fre))

    x_labels = [f'Column {i}' for i in range(len(fre))]

    # ヒストグラムをプロット
    plt.bar(x_labels, fre)
    plt.xlabel('Neuron pattern', size=15)
    plt.ylabel('Patterns', size=15)
    plt.title('Histogram of Pattern Frequencies_wa_1', size=20)
    # 横軸のラベルと位置を等間隔に設定し、フォントサイズと回転を調整
    step = 100
    plt.xticks(np.arange(1, len(x_labels), step), x_labels[::step], fontsize=10)
    plt.tight_layout()  # レイアウトを調整して、ラベルが重ならないようにする
    plt.show()



if __name__ == '__main__':
    main("2024_1028_2207")