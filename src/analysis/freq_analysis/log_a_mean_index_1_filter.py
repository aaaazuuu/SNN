import os

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, subplots

log_folder = os.path.abspath("../../../log_eva")


def main(log_time: str):
    log_folder_ = os.path.join(log_folder, log_time)
    npy_file_path_1 = os.path.join(log_folder_, "log_a1_mean_index_1.npy")
    npy_file_path_2 = os.path.join(log_folder_, "w_a1.npy")
    npy_file_path_3 = os.path.join(log_folder_, "w_a2.npy")
    npy_file_path_4 = os.path.join(log_folder_, "w_a3.npy")


    log_a1_mean_index_1: np.ndarray = np.load(npy_file_path_1)
    w_a1: np.ndarray = np.load(npy_file_path_2)
    w_a2: np.ndarray = np.load(npy_file_path_3)
    w_a3: np.ndarray = np.load(npy_file_path_4)


    # log_a1_mean_index_1 = cv2.resize(log_a1_mean_index_1, dsize=None,dst=None, interpolation=cv2.INTER_LINEAR)
    print(log_a1_mean_index_1.T.shape)
    print(w_a1.shape)

    # (log-log.min)/(log.max-log.min)正規化するか
    min_values = np.min(log_a1_mean_index_1, axis=1, keepdims=True)
    max_values = np.max(log_a1_mean_index_1, axis=1, keepdims=True)
    nor_log_2 = (log_a1_mean_index_1 - min_values) / (max_values - min_values)

    # どの範囲で正規化するか
    nor_log = np.where(log_a1_mean_index_1 == max_values, 1, 0)

    # 最大値を獲得したニューロン番号のみを取得する（フィルター（結合強度）画像獲得用）
    selected_data_1 = w_a1[:, np.argmax(log_a1_mean_index_1, axis=1)]
    selected_data_2 = w_a2[:, np.argmax(log_a1_mean_index_1, axis=1)]
    selected_data_3 = w_a3[:, np.argmax(log_a1_mean_index_1, axis=1)]

    print(f'selected_data_w_a1: {selected_data_1.shape}')
    print(f'selected_data_w_a2: {selected_data_2.shape}')
    print(f'selected_data_w_a3: {selected_data_3.shape}')

    # fig.colorbar(im, ax=ax)
    # 補助線

    minor_ticks_top = np.linspace(0, log_a1_mean_index_1.shape[0], log_a1_mean_index_1.shape[0] + 1)
    # 図のサイズ
    fig_1, axes_1 = plt.subplots(1, 3, figsize=(20, 5))
    fig_2, axes_2 = plt.subplots(1, 3, figsize=(20, 5))

    # pcolor画像表示
    axes_1[0].pcolor(log_a1_mean_index_1.T)
    axes_1[1].pcolor(nor_log_2.T)
    axes_1[2].pcolor(nor_log.T)

    axes_2[0].pcolor(selected_data_1)
    axes_2[1].pcolor(selected_data_2)
    axes_2[2].pcolor(selected_data_3)



    # 補助線用の関数
    axes_1[0].set_xticks(minor_ticks_top, minor=True)
    axes_1[1].set_xticks(minor_ticks_top, minor=True)
    axes_1[2].set_xticks(minor_ticks_top, minor=True)

    axes_2[0].set_xticks(minor_ticks_top, minor=True)
    axes_2[1].set_xticks(minor_ticks_top, minor=True)
    axes_2[2].set_xticks(minor_ticks_top, minor=True)



    # ひっくり返す
    axes_1[0].invert_yaxis()
    axes_1[1].invert_yaxis()
    axes_1[2].invert_yaxis()

    axes_2[0].invert_yaxis()
    axes_2[1].invert_yaxis()
    axes_2[2].invert_yaxis()



    axes_1[0].grid(which="both", axis="x")
    axes_1[1].grid(which="both", axis="x")
    axes_1[2].grid(which="both", axis="x")

    axes_2[0].grid(which="both", axis="x")
    axes_2[1].grid(which="both", axis="x")
    axes_2[2].grid(which="both", axis="x")


    axes_1[0].set_title('normal')
    axes_1[1].set_title('normalization')
    axes_1[2].set_title('min of max')

    axes_2[0].set_title('w_a1')
    axes_2[1].set_title('w_a2')
    axes_2[2].set_title('w_a3')


    axes_1[0].set_xlabel('Tone')
    axes_1[1].set_xlabel('Tone')
    axes_1[2].set_xlabel('Tone')

    axes_2[0].set_xlabel('Tone')
    axes_2[1].set_xlabel('Tone')
    axes_2[2].set_xlabel('Tone')


    axes_1[0].set_ylabel('neuron_number')
    axes_1[1].set_ylabel('neuron_number')
    axes_1[2].set_ylabel('neuron_number')

    # tick_labels = ['A0','B0','A1','B1','C1','D.1','F.1','A2','B2','C2','D.2','F.2','A3','B3','C3','D.3','F.3','A4','B4','C4','D.4','F.4','A5','B5','C5','D.5','F.5','A6','B6','C6','D.6','F.6','A7','B7','C7','D.7','F.7','C8'] # .は＃のこと
    #
    # tick_positions = np.linspace(0.5, 37.5, len(tick_labels))  # ラベルの数に合わせて目盛りの間隔を調整
    # axes_1[0].set_xticks(tick_positions)
    # axes_1[0].grid(False)
    # axes_1[1].set_xticks(tick_positions)
    # axes_1[1].grid(False)
    # axes_1[2].set_xticks(tick_positions)
    # axes_1[2].grid(False)
    # axes_1[0].set_xticklabels(tick_labels, fontsize=6)
    # axes_1[0].grid(False)
    # axes_1[1].set_xticklabels(tick_labels, fontsize=6)
    # axes_1[1].grid(False)
    # axes_1[2].set_xticklabels(tick_labels, fontsize=6)
    # axes_1[2].grid(False)

    # axes_2[0].set_xticks(tick_positions)
    # axes_2[0].grid(False)
    # axes_2[0].set_xticklabels(tick_labels, fontsize=6)
    # axes_2[0].grid(False)
    # axes_2[1].set_xticks(tick_positions)
    # axes_2[1].grid(False)
    # axes_2[1].set_xticklabels(tick_labels, fontsize=6)
    # axes_2[1].grid(False)
    # axes_2[2].set_xticks(tick_positions)
    # axes_2[2].grid(False)
    # axes_2[2].set_xticklabels(tick_labels, fontsize=6)
    # axes_2[2].grid(False)


    print(f'neu_num:\n {np.argmax(log_a1_mean_index_1, axis=1)}')
    print(f'max_mean_spike:\n{np.max(log_a1_mean_index_1, axis=1)}')
    print(f'min_of_max:\n{min(np.max(log_a1_mean_index_1, axis=1))}')
    plt.show()


if __name__ == '__main__':
    main("2024_1226_1544_random_std_piano_C1-C7")