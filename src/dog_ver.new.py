## 高周波数ではDOGの範囲（狭）　低周波数ではDOGの範囲（広）ver.　σ1、2の調整
import datetime
import os

import numpy as np
import librosa
import librosa.display
import cv2
import matplotlib.pyplot as plt
import scipy
from numpy.ma.core import array

from lib.snn import LIFNeuron

if __name__ == '__main__':
    fpath = "../dataset/ESC-50-master/audio/1-137-A-32.wav"
    eta = 1.0  # 学習時間
    n_fft = 1024  # FFT（高速フーリエ変換）のサイズ
    sr = 44100  # サンプリングレート　音声をデータにする時に、１秒間を何個の点に分割するかという意味　高いと滑らかな音の取得が可能
    max_fr = 22050  ######
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


    ## for dog_filter
    def _convolve2d(image, kernel):  # 入力画像に対して畳み込みフィルタ（kernel）を適用している
        shape = (
                image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1) + kernel.shape  # 出力shapeの決定
        strides = image.strides * 2  # メモリ上でのアクセスの調節 カーネルが画像内で移動しながら演算ができるようになる
        strided_image = np.lib.stride_tricks.as_strided(image, shape,
                                                        strides)  ## 配列のstrideを調節して、元の配列のデータを再構築せずに異なる形で見ることができる　imageを新しいshapeとstridesに基づいて再構築
        return np.einsum('kl,ijkl->ij', kernel, strided_image)  ## karnelとstrided_imageの要素を適切にかけ合わせて畳み込みの結果を得ることができる


    ## for dog_filter
    def _convolve2d_multichannel(image, kernel):  # カラー画像の時に使う
        convolved_image = np.empty((image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1,
                                    image.shape[2]))  # image.shape[2]は、チャネル数RGB
        for i in range(image.shape[2]):
            convolved_image[:, :, i] = _convolve2d(image[:, :, i], kernel)  # 全チャネルにカーネルを適用させる
        return convolved_image


    ## for dog_filter
    def _pad_singlechannel_image(image, kernel_shape, boundary):  # パディング設定
        return np.pad(image, ((int(kernel_shape[0] / 2),), (int(kernel_shape[1] / 2),)),
                      boundary)  # カーネルサイズの半分のサイズでパディングが追加（3×3 → 4×4）


    ## for dog_filter
    def _pad_multichannel_image(image, kernel_shape, boundary):  # カラーver.
        return np.pad(image, ((int(kernel_shape[0] / 2),), (int(kernel_shape[1] / 2),), (0,)), boundary)


    ## for dog_filter
    def convolve2d(image, kernel, boundary='edge'):  # 画像が単一チャネルかマルチチャネルかを確認してそれに応じて適切なパディングと畳み込み処理を行う
        if image.ndim == 2:
            pad_image = _pad_singlechannel_image(image, kernel.shape,
                                                 boundary) if boundary is not None else image  # boundaryがNoneの場合パディング処理なしのimageそのまま
            return _convolve2d(pad_image, kernel)
        elif image.ndim == 3:
            pad_image = _pad_multichannel_image(image, kernel.shape, boundary) if boundary is not None else image
            return _convolve2d_multichannel(pad_image, kernel)


    ## for dog_filter
    def create_gaussian_kernel(size=(5, 5), sigma=1):
        center = ((size[0] - 1) / 2, (size[1] - 1) / 2)
        sigma2 = 2 * sigma * sigma
        kernel = np.fromfunction(lambda y, x: np.exp(-((x - center[1]) ** 2 + (y - center[0]) ** 2) / sigma2),
                                 size)  # fromfunction(lambda 引数: 返り値, outputの行列の形,)
        kernel = kernel / np.sum(kernel)  # 重みの合計が１になるように正規化している
        return kernel


    # dog_filter σ可変ver
    def dog_filter(image, sigma1=2.0, sigma2=1.0, size=(7, 1), boundary='edge'):
        kernel1 = create_gaussian_kernel(size=size, sigma=sigma1)
        kernel2 = create_gaussian_kernel(size=size, sigma=sigma2)
        gauss_image1 = convolve2d(image, kernel1, boundary=boundary)
        gauss_image2 = convolve2d(image, kernel2, boundary=boundary)
        return gauss_image1 - gauss_image2


    wav, sr = librosa.load(path=fpath, sr=sr)
    csp = librosa.cqt(wav, sr=sr, hop_length=hop_length, fmin=librosa.note_to_hz('C1'), bins_per_octave=bins_per_octave, n_bins=n_bins)

    cspr = csp.real  # 複素数の実部を取得
    lcspr = np.log(cspr.clip(0, None) + 1)


    original_image = lcspr
    if np.issubdtype(original_image.dtype, np.integer):  # 指定した画像の型がinteger型（整数型の集合体）かどうかをチェックする
        original_image = original_image / np.iinfo(original_image.dtype).max  # original_imageをclip(0,1)
    normalized_original_image = (original_image * 255).astype(np.uint8)  # opencv用処理(lcspr画像と同じ)
    # print(f'original_image_max : {original_image.max()}')

    # dog_filtered_image
    dog_image = dog_filter(original_image, max_fr)
    value_range = max(abs(dog_image.min()), abs(dog_image.max()))
    normalized_dog_image = np.clip(dog_image, 0, value_range)  # opencv用処理
    normalized_dog_image = (normalized_dog_image / value_range * 255).astype(np.uint8)  # opencv用処理


    cv2.imshow('Dogfiltered_lcspr_img', normalized_dog_image)  # DOGフィルターをかけた後のlcspr画像（側方抑制機構）

    cv2.waitKey(0)

cv2.destroyAllWindows()



