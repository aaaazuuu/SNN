import datetime
import os

import numpy as np
import librosa
import librosa.display
import cv2
import matplotlib.pyplot as plt
import scipy
from numpy.core.fromnumeric import reshape
from numpy.ma.core import shape

from lib.snn import LIFNeuron
from src.cqt_practice import bins_per_octave

n_bins = 256
bins_per_octave = 12 * 3


# def im2col(input_data, filter_h, filter_w, stride=1, pad=0):  # 畳み込み用に画像データを2次元配列に変換する操作  フィルタのサイズの計算と出力サイズの計算  とりあえずフィルタを入れた時の出力サイズの計算している
#     N, C, H, W = input_data.shape  # 入力データの形状取得　N；バッチサイズ、ｃ；チャネル数、Ｈ；高さ、Ｗ；幅
#     # filter_h, filter_w, _, _ = w.shape  # フィルタの形状を取得　filter_h;フィルターの高さ、filter_w;フィルターの幅
#     out_h = (H + 2 * pad - filter_h) // stride + 1  # 出力の後の画像サイズの高さ（公式）
#     print(f"out_h : {out_h}")
#     out_w = (W + 2 * pad - filter_w) // stride + 1  # 出力の幅　計算方法（公式）
#     print(f"out_w : {out_w}")
#     img = np.pad(input_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)  # 入力データにパディングを追加した時の出力画像サイズ取得
#     print(f"{img}")
#     col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))  # 出力用の配列を初期化
#
#     for y in range(filter_h):  # 入力データのスライドウィンドウの作成　入力データimgに対してフィルターをスライドさせながら対応するcol配列に格納する
#         y_max = y + stride * out_h  # 入力画像にフィルタをかけてストライドの大きさづつスライドさせた時の出力画像サイズの最大値取得（ｈ）
#         for x in range(filter_w):
#             # ここあした岩舘先生に聞く
#             x_max = x + stride * out_w  # 入力画像にフィルタをかけてストライドの大きさづつスライドさせた時の出力画像サイズの最大値取得（w）
#             col[:, :, y, x] = img[:, :, y:y_max:stride, x:x_max:stride]
#     col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)  # col配列を転置してからreシェイプする　より、col配列は２次元の行列になり各行がフィルターの適用される領域を示す
#     #                   N  H, W, C  Fh Fw         N*H*W x C*Fh*Fw
#
#     print(col.shape)
#     return col


## for qsq_filter
def qsq_kernel(): # このフィルタ解析するときどこかのタイミングでかけたい
    n_oct = 20  # n_oct倍音まで考慮する
    ref = np.arange(1, n_oct) * librosa.note_to_hz('C1')  # ドの倍音20個分が出力される
    cqt_freqs = librosa.cqt_frequencies(n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=librosa.note_to_hz('C1'))
    dif = (ref.reshape(-1, 1) - cqt_freqs.reshape(1, -1)) ** 2
    difmax = dif.argmin(axis=1).max()
    print('difmax', dif.argmin(axis=1).max())
    mask = np.zeros(difmax + 1)
    mask[dif.argmin(axis=1)] = 1.0
    print(mask.argmin(axis=0).max())
    return cv2.resize(mask[::-1], None, fx=10, fy=1)


## for qsq_filter
def _convolve2d(image, kernel):  # 入力画像に対して畳み込みフィルタ（kernel）を適用している
    shape = (image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1) + kernel.shape  # 出力shapeの決定
    strides = image.strides * 2  # メモリ上でのアクセスの調節 カーネルが画像内で移動しながら演算ができるようになる
    strided_image = np.lib.stride_tricks.as_strided(image, shape, strides)  ## 配列のstrideを調節して、元の配列のデータを再構築せずに異なる形で見ることができる　imageを新しいshapeとstridesに基づいて再構築
    return np.einsum('kl,ijkl->ij', kernel, strided_image)  ## karnelとstrided_imageの要素を適切にかけ合わせて畳み込みの結果を得ることができる


## for qsq_filter
def _convolve2d_multichannel(image, kernel):  # カラー画像の時に使う
    convolved_image = np.empty((image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1, image.shape[2]))  # image.shape[2]は、チャネル数RGB
    for i in range(image.shape[2]):
        convolved_image[:, :, i] = _convolve2d(image[:, :, i], kernel)  # 全チャネルにカーネルを適用させる
    return convolved_image


## for qsq_filter
def _pad_singlechannel_image(image, kernel_shape, boundary):  # パディング設定
    return np.pad(image, ((int(kernel_shape[0] / 2),), (int(kernel_shape[1] / 2),)), boundary)  # カーネルサイズの半分のサイズでパディングが追加（3×3 → 4×4）


## for qsq_filter
def _pad_multichannel_image(image, kernel_shape, boundary):  # カラーver.
    return np.pad(image, ((int(kernel_shape[0] / 2),), (int(kernel_shape[1] / 2),), (0,)), boundary)


## for qsq_filter
def convolve2d(image, kernel, boundary='None'):  # 画像が単一チャネルかマルチチャネルかを確認してそれに応じて適切なパディングと畳み込み処理を行う
    if image.ndim == 2:
        pad_image = _pad_singlechannel_image(image, kernel.shape, boundary) if boundary is not None else image  # boundaryがNoneの場合パディング処理なしのimageそのまま
        return _convolve2d(pad_image, kernel)
    elif image.ndim == 3:
        pad_image = _pad_multichannel_image(image, kernel.shape, boundary) if boundary is not None else image
        return _convolve2d_multichannel(pad_image, kernel)


## for qsq_filter
def create_gaussian_kernel(size=(5, 5), sigma=1):
    center = ((size[0] - 1) / 2, (size[1] - 1) / 2)
    sigma2 = 2 * sigma * sigma
    kernel = np.fromfunction(lambda y, x: np.exp(-((x - center[1]) ** 2 + (y - center[0]) ** 2) / sigma2), size)  # fromfunction(lambda 引数: 返り値, outputの行列の形,)
    kernel = kernel / np.sum(kernel)  # 重みの合計が１になるように正規化している
    return kernel

## for qsq_filter
def qsq_filter(image, size=(7, 1)):
    kernel = qsq_kernel()
    qsq_img =
    return qsq_img

if __name__ == '__main__':
    fpath = "../dataset/ESC-50-master/audio/1-137-A-32.wav"
    eta = 1.0  # 学習時間
    n_fft = 1024  # FFT（高速フーリエ変換）のサイズ
    sr = 44100  # サンプリングレート　音声をデータにする時に、１秒間を何個の点に分割するかという意味　高いと滑らかな音の取得が可能
    poisson = True
    n_bins = 256
    hop_length = 192
    bins_per_octave = 12 * 3
    epoch_count = 0
    lateral = True
    load = False
    draw = True

    exc_tau = 1.0
    exc_scale = 1.6

    inh_tau = 1.0
    inh_scale = 1.6

    n_a1 = 340
    n_a2 = 640
    n_a3 = 750  # 適当

    # decay 1:on / 0:off
    cell_type = 0
    substep = 2

    tgan = 0.0
    rgan = 0.0

    if cell_type == 1:
        tgan = 1.0
    rgan = 0.05

    wav, sr = librosa.load(path=fpath, sr=sr)
    csp = librosa.cqt(wav, sr=sr, hop_length=hop_length, fmin=librosa.note_to_hz('C1'), bins_per_octave=bins_per_octave, n_bins=n_bins)
    cspr = csp.real
    cspr = cspr.reshape(1, 1, 256, -1)
    print(f"csp.shape : {cspr.shape}")
    x = np.ones((1, 1, 5, 5))

    col = im2col(x, filter_h=2, filter_w=2, pad=1)

    image = cv2.imread('cqt_spectrum.png')

    cv2.imshow('Constant-Q power spectrum', image)
    cv2.imshow('csp', lcspr / lcspr.max())  # CQT変換して正規化してる
    # print(f'lcspr_max : {lcspr.max()}')

    # original_image  lcspr画像(複素数データの実部を取得し、対数変換を行いデータのスケールを圧縮したもの)
    original_image = lcspr
    if np.issubdtype(original_image.dtype, np.integer):  # 指定した画像の型がinteger型（整数型の集合体）かどうかをチェックする
        original_image = original_image / np.iinfo(original_image.dtype).max  # original_imageをclip(0,1)
    normalized_original_image = (original_image * 255).astype(np.uint8)  # opencv用処理(lcspr画像と同じ)
    # print(f'original_image_max : {original_image.max()}')

    # dog_filtered_image
    dog_image = qsq_filter(original_image)
    value_range = max(abs(dog_image.min()), abs(dog_image.max()))
    normalized_dog_image = np.clip(dog_image, 0, value_range)  # opencv用処理
    normalized_dog_image = (normalized_dog_image / value_range * 255).astype(np.uint8)  # opencv用処理
    # print(f'normalized_dog_image_max : {normalized_dog_image.max()}')


cv2.imshow("qsq", mask)

cv2.waitKey(0)

cv2.destroyAllWindows()