import os

import numpy as np

import matplotlib as plt

from src.analysis.w_a1_pattern_freq import log_folder

def main(log_time: str):
    log_folder_ = os.path.join(log_folder, log_time)
    npy_file_path = os.path.join(log_folder_, "w_a1_cossim.npy")

    w_a1_cossim: np.ndarray = np.load(npy_file_path)

    print(w_a1_cossim)


if __name__ == '__main__':
    main("2024_1029_1813")
