import os

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import librosa
import librosa.display

from matplotlib.pyplot import imshow, subplots

log_folder = os.path.abspath("../../../log_evaluation") # 評価用のデータで見ている

def main(log_time: str):


    # npファイルの取得
    log_folder_ = os.path.join(log_folder, log_time)
    npy_file_path_1 = os.path.join(log_folder_, "log_a1_mean_index_1.npy")  # "log_a1_mean_index_1"はスパイクデータの時間方向で平均を取ったやつ（データセットのファイル数分取ってスタックしている）
    npy_file_path_2 = os.path.join(log_folder_, "log_a2_mean_index_1.npy")
    npy_file_path_3 = os.path.join(log_folder_, "log_a3_mean_index_1.npy")
    npy_file_path_4 = os.path.join(log_folder_, "w_a1.npy")
    npy_file_path_5 = os.path.join(log_folder_, "w_a2.npy")
    npy_file_path_6 = os.path.join(log_folder_, "w_a3.npy")

    log_a1_mean_index_1: np.ndarray = np.load(npy_file_path_1)
    log_a2_mean_index_1: np.ndarray = np.load(npy_file_path_2)
    log_a3_mean_index_1: np.ndarray = np.load(npy_file_path_3)

    w_a1: np.ndarray = np.load(npy_file_path_4)
    w_a2: np.ndarray = np.load(npy_file_path_5)
    w_a3: np.ndarray = np.load(npy_file_path_6)

    print(f'log_a1_mean_index_1{log_a1_mean_index_1.shape}')



    # log_a1_mean_index_1 = cv2.resize(log_a1_mean_index_1, dsize=None,dst=None, interpolation=cv2.INTER_LINEAR)
    print(f'w_a1.shape: {w_a1.shape}')
    print(f'w_a2.shape: {w_a2.shape}')
    print(f'w_a3.shape: {w_a3.shape}')


    # 最大値を獲得したニューロン番号の結合強度を取得する（フィルター（結合強度）画像獲得用）
    selected_data_1 = w_a1[:, np.argmax(log_a1_mean_index_1, axis=1)]
    selected_data_2 = w_a2[:, np.argmax(log_a2_mean_index_1, axis=1)]
    selected_data_3 = w_a3[:, np.argmax(log_a3_mean_index_1, axis=1)]

    # selected_data_1/sum
    sum_values_1 = np.sum(selected_data_1, axis=0, keepdims=True)
    sum_values_2 = np.sum(selected_data_2, axis=0, keepdims=True)
    sum_values_3 = np.sum(selected_data_3, axis=0, keepdims=True)

    log_nor_1 = selected_data_1 / sum_values_1
    log_nor_2 = selected_data_2 / sum_values_2
    log_nor_3 = selected_data_3 / sum_values_3

    print(f'log_nor_1.shape: {log_nor_1.shape}')
    print(f'log_nor_2.shape: {log_nor_2.shape}')
    print(f'log_nor_3.shape: {log_nor_3.shape}')


    # 各トーンごとに掛ける
    all_tones_1 = []
    all_tones_2 = []
    all_tones_3 = []

    for i in range(log_nor_1.shape[1]):
        get_one_tone = log_nor_1[:, i]
        get_one_tone_reshaped = get_one_tone[:, None]
        one_tone_times_w_1 = np.multiply(get_one_tone_reshaped, w_a1)
        one_tone_sum = np.sum(one_tone_times_w_1, axis=1)
        all_tones_1.append(one_tone_sum)

    all_tone_1 = np.array(all_tones_1)
    print(f'all_tone_1.shape: {all_tone_1.shape}')

    for i in range(log_nor_2.shape[1]):
        get_one_tone = log_nor_2[:, i]
        get_one_tone_reshaped = get_one_tone[:, None]
        one_tone_times_w_2 = np.multiply(get_one_tone_reshaped, w_a2)
        one_tone_sum = np.sum(one_tone_times_w_2, axis=1)
        all_tones_2.append(one_tone_sum)

    all_tone_2 = np.array(all_tones_2)
    print(f'all_tone_2.shape: {all_tone_2.shape}')


    for i in range(log_nor_3.shape[1]):
        get_one_tone = log_nor_3[:, i]
        get_one_tone_reshaped = get_one_tone[:, None]
        one_tone_times_w_3 = np.multiply(get_one_tone_reshaped, w_a3)
        one_tone_sum = np.sum(one_tone_times_w_3, axis=1)
        all_tones_3.append(one_tone_sum)

    all_tone_3 = np.array(all_tones_3)
    print(f'all_tone_3.shape: {all_tone_3.shape}')


    # fig.colorbar(im, ax=ax)
    # 補助線
    minor_ticks_top = np.linspace(0, log_a1_mean_index_1.shape[0], log_a1_mean_index_1.shape[0] + 1)
    # 図のサイズ
    fig_1, axes_1 = plt.subplots(1, 3, figsize=(20, 5))
    fig_2, axes_2 = plt.subplots(1, 3, figsize=(20, 5))
    fig_3, axes_3 = plt.subplots(1, 3, figsize=(20, 5))


    # pcolor画像表示
    axes_1[0].pcolor(all_tone_1.T)
    axes_1[1].pcolor(all_tone_2.T)
    axes_1[2].pcolor(all_tone_3.T)

    axes_2[0].pcolor(selected_data_1)
    axes_2[1].pcolor(selected_data_2)
    axes_2[2].pcolor(selected_data_3)


    # librosaを用いたy軸ラベル表示
    # 対応する周波数軸を生成 (最低周波数: 20Hz)
    n_bins = all_tone_1.T.shape[0]
    frequencies = librosa.cqt_frequencies(n_bins, fmin=librosa.note_to_hz('C1'), bins_per_octave=12*3)
    # 周波数を音階に変換
    notes = [librosa.hz_to_note(f) for f in frequencies]
    # 可視化
    pcm = axes_3[0].pcolor(all_tone_1.T, cmap='viridis')
    # Y軸を周波数に設定
    num_labels = 84  # 表示するラベルの数
    tick_positions = np.linspace(0, n_bins - 1, num_labels).astype(int)
    axes_3[0].set_yticks(tick_positions)  # 適切な間隔でインデックスを指定
    axes_3[0].set_yticklabels([notes[i] for i in tick_positions], fontsize=6)  # 周波数ラベル


    # # 補助線用の関数
    axes_1[0].set_xticks(minor_ticks_top, minor=True)
    axes_1[1].set_xticks(minor_ticks_top, minor=True)
    axes_1[2].set_xticks(minor_ticks_top, minor=True)


    axes_2[0].set_xticks(minor_ticks_top, minor=True)
    axes_2[1].set_xticks(minor_ticks_top, minor=True)
    axes_2[2].set_xticks(minor_ticks_top, minor=True)

    # axes_3[0].set_xticks(minor_ticks_top, minor=True)
    # axes_3[1].set_xticks(minor_ticks_top, minor=True)
    # axes_3[2].set_xticks(minor_ticks_top, minor=True)

    # # ひっくり返す
    axes_1[0].invert_yaxis()
    axes_1[1].invert_yaxis()
    axes_1[2].invert_yaxis()

    axes_2[0].invert_yaxis()
    axes_2[1].invert_yaxis()
    axes_2[2].invert_yaxis()

    axes_3[0].invert_yaxis()
    axes_3[1].invert_yaxis()
    axes_3[2].invert_yaxis()



    axes_1[0].grid(which="both", axis="x")
    axes_1[1].grid(which="both", axis="x")
    axes_1[2].grid(which="both", axis="x")

    axes_2[0].grid(which="both", axis="x")
    axes_2[1].grid(which="both", axis="x")
    axes_2[2].grid(which="both", axis="x")

    # axes_3[0].grid(which="both", axis="x")
    # axes_3[1].grid(which="both", axis="x")
    # axes_3[2].grid(which="both", axis="x")


    axes_1[0].set_title('a1')
    axes_1[1].set_title('a2')
    axes_1[2].set_title('a3')

    axes_2[0].set_title('w_a1')
    axes_2[1].set_title('w_a2')
    axes_2[2].set_title('w_a3')

    axes_3[0].set_title('w_a1')
    axes_3[1].set_title('w_a2')
    axes_3[2].set_title('w_a3')


    axes_1[0].set_xlabel('Tone')
    axes_1[1].set_xlabel('Tone')
    axes_1[2].set_xlabel('Tone')

    axes_2[0].set_xlabel('Tone')
    axes_2[1].set_xlabel('Tone')
    axes_2[2].set_xlabel('Tone')

    axes_3[0].set_xlabel('Tone')
    axes_3[1].set_xlabel('Tone')
    axes_3[2].set_xlabel('Tone')


    axes_1[0].set_ylabel('f')
    axes_1[1].set_ylabel('f')
    axes_1[2].set_ylabel('f')

    # tick_labels = ['C1','C#1','D1','D#1','E1','F1','F#1','G1','G#1','A1','A#1','B1','C2','C#2','D2','D#2','E2','F2','F#2','G2','G#2','A2','A#2','B2','C3','C#3','D3','D#3','E3','F3','F#3','G3','G#3','A3','A#3','C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4','C5','C#5','D5','D#5','E5','F5','F#5','G5','G#5','A5','A#5','B5','C6','C#6','D6','D#6','E6','F6','F#6','G6','G#6','A6','A#6','B6','C7','C#7','D7','D#7','E7','F7','F#7','G7','G#7','A7','A#7','B7','C8','C#8'] # .は＃のこと
    #
    # print(len(tick_labels))
    # tick_positions = np.linspace(0.5, 84.5, len(tick_labels))  # ラベルの数に合わせて目盛りの間隔を調整
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

    # axes_3[0].set_xticks(tick_positions)
    # axes_3[0].grid(False)
    # axes_3[1].set_xticks(tick_positions)
    # axes_3[1].grid(False)
    # axes_3[2].set_xticks(tick_positions)
    # axes_3[2].grid(False)
    # axes_3[0].set_xticklabels(tick_labels, fontsize=6)
    # axes_3[0].grid(False)
    # axes_3[1].set_xticklabels(tick_labels, fontsize=6)
    # axes_3[1].grid(False)
    # axes_3[2].set_xticklabels(tick_labels, fontsize=6)
    # axes_3[2].grid(False)

    plt.show()


if __name__ == '__main__':
    main("2025_0115_1128_only_piano_std_piano_C1-C7")
