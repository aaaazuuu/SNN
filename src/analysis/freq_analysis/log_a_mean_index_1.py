import os
import yaml
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, subplots

log_folder = os.path.abspath("../../../log")

from src.analysis.freq_analysis.log_a_mean_index_1 import log_folder





def main(log_time: str):
    log_folder_ = os.path.join(log_folder, log_time)
    npy_file_path = os.path.join(log_folder_, "log_a3_mean_index_1.npy")
    log_a3_mean_index_1: np.ndarray = np.load(npy_file_path)

    # log_a1_mean_index_1 = cv2.resize(log_a1_mean_index_1, dsize=None,dst=None, interpolation=cv2.INTER_LINEAR)
    print(log_a3_mean_index_1.T.shape)

    # (log-log.min)/(log.max-log.min)正規化するか
    min_values = np.min(log_a3_mean_index_1, axis=1, keepdims=True)
    max_values = np.max(log_a3_mean_index_1, axis=1, keepdims=True)
    nor_log_2 = (log_a3_mean_index_1 - min_values) / (max_values - min_values)

    # どの範囲で正規化するか
    nor_log = np.where(log_a3_mean_index_1 >= 0.0386, 1, 0)

    # fig.colorbar(im, ax=ax)
    # 補助線

    minor_ticks_top = np.linspace(0, log_a3_mean_index_1.shape[0], log_a3_mean_index_1.shape[0] + 1)
    # 図のサイズ
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    # pcolor画像表示
    axes[0].pcolor(log_a3_mean_index_1.T)
    axes[1].pcolor(nor_log_2.T)
    axes[2].pcolor(nor_log.T)

    # 補助線用の関数
    axes[0].set_xticks(minor_ticks_top, minor=True)
    axes[1].set_xticks(minor_ticks_top, minor=True)
    axes[2].set_xticks(minor_ticks_top, minor=True)

    # ひっくり返す
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    axes[2].invert_yaxis()

    axes[0].grid(which="both", axis="x")
    axes[1].grid(which="both", axis="x")
    axes[2].grid(which="both", axis="x")

    axes[0].set_title('normal')
    axes[1].set_title('normalization')
    axes[2].set_title('min of max')

    axes[0].set_xlabel('Tone')
    axes[1].set_xlabel('Tone')
    axes[2].set_xlabel('Tone')

    axes[0].set_ylabel('neuron_number')
    axes[1].set_ylabel('neuron_number')
    axes[2].set_ylabel('neuron_number')

    tick_labels = ['A0','A1','A2','A3','A4','A5','A6','A7','B1','B2','B3','B4','B5','B6','B7','C1','C2','C3','C4','C5','C6','C7','C8','D.1','D.2','D.3','D.4','D.5','D.6','D.7','F.1','F.2','F.3','F.4','F.5','F.6','F.7','C8'] # .は＃のこと

    tick_positions = np.linspace(0.5, 37.5, len(tick_labels))  # ラベルの数に合わせて目盛りの間隔を調整
    axes[0].set_xticks(tick_positions)
    axes[0].grid(False)
    axes[1].set_xticks(tick_positions)
    axes[1].grid(False)
    axes[2].set_xticks(tick_positions)
    axes[2].grid(False)
    axes[0].set_xticklabels(tick_labels, fontsize=6)
    axes[0].grid(False)
    axes[1].set_xticklabels(tick_labels, fontsize=6)
    axes[1].grid(False)
    axes[2].set_xticklabels(tick_labels, fontsize=6)
    axes[2].grid(False)



    print(f'neu_num:\n {np.argmax(log_a3_mean_index_1, axis=1)}')
    print(f'max_mean_spike:\n{np.max(log_a3_mean_index_1, axis=1)}')
    print(f'min_of_max:\n{min(np.max(log_a3_mean_index_1, axis=1))}')
    plt.show()


if __name__ == '__main__':
    main("2024_1117_1753_after_2014-2051st_std_piano_A0-F0")

