import os
import numpy as np

# 学習後のw（w_a1,w_a2,w_a3）を使用できるようにする
log_folder = os.path.abspath("../log")  # logフォルダにアクセスする（相対パス→絶対パス）学習後wを参照する為のパス
log_time = '2025_0113_1733_studied_only_piano_inp' #

# log_folderとlog_timeを結合して特定のディレクトリにアクセス
saved_folder = os.path.join(log_folder, log_time)

# w_a1,w_a2,w_a3というファイル名のnpファイルパスにアクセス
npy_file_path_1 = os.path.join(saved_folder, "w_a1.npy")
npy_file_path_2 = os.path.join(saved_folder, "w_a2.npy")
npy_file_path_3 = os.path.join(saved_folder, "w_a3.npy")

# アクセスしたファイルパスのデータを、w_a1,w_a2,w_a3という変数名で使えるようにする
w_a1: np.ndarray = np.load(npy_file_path_1)
w_a2: np.ndarray = np.load(npy_file_path_2)
w_a3: np.ndarray = np.load(npy_file_path_3)

w_a1 = w_a1
w_a2 = w_a2
w_a3 = w_a3