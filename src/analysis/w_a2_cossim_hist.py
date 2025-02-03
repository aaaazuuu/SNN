import os

import numpy as np

import matplotlib.pyplot as plt


log_folder = os.path.abspath("../../log")


def main(log_time: str):
    # pathの作成
    log_folder_ = os.path.join(log_folder, log_time)
    npy_file_path = os.path.join(log_folder_, "w_a2_cossim.npy")


    w_a2_cossim: np.ndarray = np.load(npy_file_path)
    print(w_a2_cossim.shape)

    print(w_a2_cossim)


    plt.figure(figsize=(10, 6))
    plt.hist(w_a2_cossim, bins=30, alpha=0.7, color='blue', edgecolor='black')

    # ラベルとタイトルを設定
    plt.title('Histogram of Cosine Similarities_w_a2')
    plt.xlabel('Cossim Similarity')
    plt.ylabel('Frequency')

    # 表示
    plt.xlim(0, 0.4)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

if __name__ == '__main__':
    main("2024_1029_1813")