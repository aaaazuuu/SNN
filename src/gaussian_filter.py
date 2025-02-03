import matplotlib.pyplot as plt
import numpy as np
import cv2

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

def create_averaging_kernel(size = (5, 5)):
    return np.full(size, 1 / (size[0] * size[1]))  # 1 / カーネルの全要素数

# gaussian_filter
def create_gaussian_kernel(size=(5, 5), sigma=1):
    center = ((size[0] - 1) / 2, (size[1] - 1) / 2)
    sigma2 = 2 * sigma * sigma
    kernel = np.fromfunction(lambda y, x: np.exp(-((x - center[1]) ** 2 + (y - center[0]) ** 2) / sigma2), size)  # fromfunction(lambda 引数: 返り値, outputの行列の形,)
    kernel = kernel / np.sum(kernel)  # 重みの合計が１になるように正規化している
    return kernel

# original
original_image = plt.imread('C:/workspace/SNN/src/figs/img_test.jpg')
if np.issubdtype(original_image.dtype, np.integer):
    original_image = original_image / np.iinfo(original_image.dtype).max
plt.figure(figsize = (10, 8))
plt.subplot(2, 2, 1)
plt.imshow(original_image)

# # 3×3平均化フィルター
# averaging_kernel_3_3 = create_averaging_kernel(size=(3, 3))
# averaging_image_3_3 = convolve2d(original_image, averaging_kernel_3_3)
# plt.subplot(3, 2, 2)
# plt.imshow(averaging_image_3_3)

# # 5×5平均化フィルター
# averaging_kernel_5_5 = create_averaging_kernel()
# averaging_image_5_5 = convolve2d(original_image, averaging_kernel_5_5)
# plt.subplot(3, 2, 3)
# plt.imshow(averaging_image_5_5)
#
# # 11×11平均化フィルター
# averaging_kernel_11_11 = create_averaging_kernel(size=(11, 11))
# averaging_image_11_11 = convolve2d(original_image, averaging_kernel_11_11)
# plt.subplot(3, 2, 4)
# plt.imshow(averaging_image_11_11)
#
# # 縦方向のみ平均化フィルター
# averaging_kernel_17_1 = create_averaging_kernel(size=(17, 1))
# averaging_image_11_11 = convolve2d(original_image, averaging_kernel_17_1)
# plt.subplot(3, 2, 5)
# plt.imshow(averaging_image_11_11)

# gaussianフィルター
gaussian_kernel1 = create_gaussian_kernel()
gaussian_image = convolve2d(original_image, gaussian_kernel1)
plt.subplot(2, 2, 2)
plt.imshow(gaussian_image)

# gaussianフィルター2
gaussian_kernel2 = create_gaussian_kernel(size=(9, 9), sigma=3)
gaussian_image2 = convolve2d(original_image, gaussian_kernel2)
plt.subplot(2, 2, 3)
plt.imshow(gaussian_image2)
plt.show()

