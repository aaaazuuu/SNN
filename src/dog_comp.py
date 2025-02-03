import matplotlib.pyplot as plt
import numpy as np

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

def create_gaussian_kernel(size=(5, 5), sigma=1):
    center = ((size[0] - 1) / 2, (size[1] - 1) / 2)
    sigma2 = 2 * sigma * sigma
    kernel = np.fromfunction(lambda y, x: np.exp(-((x - center[1]) ** 2 + (y - center[0]) ** 2) / sigma2), size)  # fromfunction(lambda 引数: 返り値, outputの行列の形,)
    kernel = kernel / np.sum(kernel)  # 重みの合計が１になるように正規化している
    return kernel

# dog_filter
def dog_filter(image, sigma1=2.0, sigma2=1.0, size=(7, 1), boundary='edge'):
    kernel1 = create_gaussian_kernel(size=size, sigma=sigma1)
    kernel2 = create_gaussian_kernel(size=size, sigma=sigma2)
    gauss_image1 = convolve2d(image, kernel1, boundary=boundary)
    gauss_image2 = convolve2d(image, kernel2, boundary=boundary)
    return gauss_image1 - gauss_image2

original_image = plt.imread('C:/workspace/SNN/src/figs/img_test.jpg')
if np.issubdtype(original_image.dtype, np.integer):  # 指定した画像の型がinteger型（整数型の集合体）かどうかをチェックする
    original_image = original_image / np.iinfo(original_image.dtype).max
gray_image = 0.2116 * original_image[:,:,0] + 0.7152 * original_image[:,:,1] + 0.0722 * original_image[:,:,2]

# dog_filter
dog_image = dog_filter(gray_image)
value_range = max(abs(dog_image.min()), abs(dog_image.max()))

# filter_img
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
kernel1 = create_gaussian_kernel(size=(7, 1), sigma=2.0)
plt.imshow(kernel1, cmap='gray')
plt.subplot(1, 2, 2)
kernel2 = create_gaussian_kernel(size=(7, 1), sigma=1.0)
plt.imshow(kernel2, cmap='gray')

# figs
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.subplot(2, 2, 2)
plt.imshow(dog_image, cmap='gray', vmin=-value_range, vmax=value_range)
plt.colorbar()
plt.show()