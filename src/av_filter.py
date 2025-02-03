import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy.ma.extras import average


def _convolve2d(image, kernel):  # 入力画像に対して畳み込みフィルタ（kernel）を適用している
    shape = (image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1) + kernel.shape  # 出力shapeの決定
    strides = image.strides * 2  # メモリ上でのアクセスの調節 カーネルが画像内で移動しながら演算ができるようになる
    strided_image = np.lib.stride_tricks.as_strided(image, shape, strides)  ## 配列のstrideを調節して、元の配列のデータを再構築せずに異なる形で見ることができる　imageを新しいshapeとstridesに基づいて再構築
    return np.einsum('kl,ijkl->ij', kernel, strided_image)  ## karnelとstrided_imageの要素を適切にかけ合わせて畳み込みの結果を得ることができる

def _convolve2d_multichannel(image, kernel):  # カラー画像の時に使う
    convolved_image = np.empty((image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1, image.shape[2]))  # image.shape[2]は、チャネル数RGB
    for i in range(image.shape[2]):
        convolved_image[:,:,i] = _convolve2d(image[:,:,i], kernel)  # 全チャネルにカーネルを適用させる
    return convolved_image

def _pad_singlechannel_image(image, kernel_shape, boundary):  # パディング設定
    return np.pad(image, ((int(kernel_shape[0] / 2),), (int(kernel_shape[1] / 2),)), boundary)  # カーネルサイズの半分のサイズでパディングが追加（3×3 → 4×4）

def _pad_multichannel_image(image, kernel_shape, boundary):  # カラーver.
    return  np.pad(image, ((int(kernel_shape[0] / 2),), (int(kernel_shape[1] / 2),), (0,)), boundary)

def convolve2d(image, kernel, boundary='edge'):  # 画像が単一チャネルかマルチチャネルかを確認してそれに応じて適切なパディングと畳み込み処理を行う
    if image.ndim == 2:
        pad_image = _pad_singlechannel_image(image, kernel.shape, boundary) if boundary is not None else image  # boundaryがNoneの場合パディング処理なしのimageそのまま
        return _convolve2d(pad_image, kernel)
    elif image.ndim == 3:
        pad_image = _pad_multichannel_image(image, kernel.shape, boundary) if boundary is not None else image
        return _convolve2d_multichannel(pad_image, kernel)

original_image = cv2.imread('C:/workspace/SNN/src/figs/img_test.jpg')
if np.issubdtype(original_image.dtype, np.integer):  # もし整数型であればその画素値を0から1にスケーリングすることでmatplotの表示範囲にする
    original_image = original_image / np.iinfo(original_image.dtype).max  # 0～1にスケーリングする


# 3×3の平均化フィルター
def create_averaging_kernel(size):
    return np.full(size, 1 / (size[0] * size[1]))

average3_3_kernel = create_averaging_kernel((3, 3))
average3_3_image = convolve2d(original_image, average3_3_kernel)

# 5×5の平均化フィルター
def create_averaging_kernel(size):
    return np.full(size, 1 / (size[0] * size[1]))

average5_5_kernel = create_averaging_kernel((5, 5))
average5_5_image = convolve2d(original_image, average5_5_kernel)

# sobelフィルター 出力までは実装していない　とりあえず書いた
def sobel_filter():
    gray_image = 0.2116 * original_image[:, :, 0] + 0.7157 * original_image[:, :, 1] + 0.0722 * original_image[:, :, 2]

    sobel_h_kernel = np.array(([1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]))
    sobel_h_image = convolve2d(gray_image, sobel_h_kernel)
    value_range = max(abs(sobel_h_image.min()), abs(sobel_h_image.max()))
    plt.imshow(sobel_h_image, cmap='bwr', vmin=-value_range, vmax=value_range)
    plt.colorbar()


cv2.imshow('original_image', original_image)
cv2.imshow('average3_3_image', average3_3_image)
cv2.imshow('average5_5_image', average5_5_image)

cv2.waitKey(0)

cv2.destroyAllWindows()



