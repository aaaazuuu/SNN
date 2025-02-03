import os

import numpy as np

import matplotlib.pyplot as plt

from src.analysis.w_a1_pattern_freq import log_folder

def main(log_time: str):
    log_folder_ = os.path.join(log_folder, log_time)
    npy_file_path = os.path.join(log_folder_, "w_a1_cossim_num_check.npy")

    w_a1_cossim_num_check: np.ndarray = np.load(npy_file_path)

    print(w_a1_cossim_num_check)

    plt.scatter(w_a1_cossim_num_check[0], w_a1_cossim_num_check[1])
    plt.show()


if __name__ == '__main__':
    main("2024_1108_1332")