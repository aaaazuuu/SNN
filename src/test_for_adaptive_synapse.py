import datetime
import os

import numpy as np
import librosa
import librosa.display
import cv2
import matplotlib.pyplot as plt
import scipy
from numpy.core.shape_base import vstack

from numpy.ma.core import array
from numpy.lib.twodim_base import flipud, fliplr

from lib.snn import LIFNeuron

if __name__ == '__main__':
    # 今の日付・時刻を取得（datetime.datetime.now()），文字列に変換する（.strftime("%Y_%m%d_%H%M")）
    now = datetime.datetime.now().strftime("%Y_%m%d_%H%M")

    log_path = f'../log/{now}/'
    fig_path = f'../fig/{now}/'
    dataset_path = '../dataset/all/'

    # dataset_path = '../voice_conversion/dataset/jvs_ver1/'
    os.makedirs(log_path, exist_ok=True)  # ログを保存するディレクトリのパス　ディレクトリがすでに存在してもエラーが出ないようにしている
    os.makedirs(fig_path, exist_ok=True)

    eta = 1.0  # 学習時間
    n_fft = 1024  # FFT（高速フーリエ変換）のサイズ
    sr = 44100  # サンプリングレート　音声をデータにする時に、１秒間を何個の点に分割するかという意味　高いと滑らかな音の取得が可能
    poisson = True
    n_bins = 256
    hop_length = 192
    bins_per_octave = 12 * 3 # 1オクターブを36周波数分に分ける
    epoch_count = -1

    tau = 0.0  # シナプスの減衰速度　///////
    rec = 0.5  # リカバリー係数　リソースが回復する速さ制御  ///////
    N = 200 # シナプスの数  ///////

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
    substep = 2 # //////////

    tgan = 0.0
    rgan = 0.0

    if cell_type == 1:
        tgan = 1.0
    rgan = 0.05

    def stdp_data(exp_stdp=True, exc_scale=1.0, inh_scale=1.0, inh_width=1, exc_tau=1.0, inh_tau=1.0):  # exp;興奮　int;抑制性 (In)
        if exp_stdp:
            stdp_width = int(max(1 / exc_tau, 1 / inh_tau) * 20)

            ltp_exc = np.exp(-np.arange(stdp_width) * exc_tau)  # 長期増強プロファイルを指数関数で表示 # 興奮性シナプスLTPカーネル
            ltd_exc = np.exp(-(np.arange(stdp_width) - 1) * exc_tau) * exc_scale  # 長期抑制プロファイルを指数関数で表示 # 興奮性シナプスLTDカーネル　
            ltd_exc[0] = 0  # 抑制側の方の初めの値は０

            ltp_inh = -np.exp(-(np.arange(stdp_width) - 1 - inh_width) * inh_tau)
            ltp_inh[0:1 + inh_width] = 0
            ltd_inh = np.exp(-(np.arange(stdp_width) - 1 - inh_width) * inh_tau)
            ltd_inh[0:1 + inh_width] = 0

            ltp_inh *= inh_scale
            ltd_inh *= inh_scale

            ltp = ltp_exc + ltp_inh
            ltd = ltd_exc + ltd_inh

            stdp = np.zeros(2 * stdp_width - 1)
            stdp[:stdp_width] = -ltd[::-1]
            stdp[stdp_width - 1:] = ltp

            plt.plot(stdp)
            plt.show()
        else:
            stdp_width = inh_width + 3

            ltp_exc = np.zeros(stdp_width)
            ltp_exc[0] = 1.0
            ltd_exc = np.zeros(stdp_width)
            ltd_exc[1] = exc_scale

            ltp_inh = np.zeros(stdp_width)
            ltp_inh[inh_width + 1] = -inh_scale
            ltd_inh = np.zeros(stdp_width)
            ltd_inh[inh_width + 1] = inh_scale

            ltp = ltp_exc + ltp_inh
            ltd = ltd_exc + ltd_inh

            stdp = np.zeros(2 * stdp_width - 1)
            stdp[:stdp_width] = -ltd[::-1]
            stdp[stdp_width - 1:] = ltp

            plt.plot(stdp)
            plt.show()

        return ltp, ltd, stdp_width


    ltp, ltd, cmax = stdp_data(exc_scale=exc_scale, inh_scale=inh_scale, exc_tau=exc_tau, inh_tau=inh_tau)

    def stdp(sin, cin, sout, cout):  # (In)　スパイクのin カウントのin スパイクのout　カウントのout
        """

        :param sin: A spike
        :param cin: A count
        :param sout: B spike
        :param cout: B count
        :return:
        """
        ltp_in = ltp[cin.astype(np.int32)]
        ltd_out = ltd[cout.astype(np.int32)]

        exc = ltp_in[:, None] @ sout[None]
        inh = sin[:, None] @ ltd_out[None]
        return exc - inh

    ## for dog_filter
    def _convolve2d(image, kernel):  # 入力画像に対して畳み込みフィルタ（kernel）を適用している
        shape = (image.shape[0] - kernel.shape[0] + 1,
                 image.shape[1] - kernel.shape[1] + 1) + kernel.shape  # 出力shapeの決定
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

    # dog_filter
    def dog_filter(image, sigma1=2.0, sigma2=1.0, size=(7, 1),
                   boundary='edge'):  # edge;境界付近でデータが不足した場合、最も近い端の値をコピーして計算に使用
        kernel1 = create_gaussian_kernel(size=size, sigma=sigma1)
        kernel2 = create_gaussian_kernel(size=size, sigma=sigma2)
        gauss_image1 = convolve2d(image, kernel1, boundary=boundary)
        gauss_image2 = convolve2d(image, kernel2, boundary=boundary)
        return gauss_image1 - gauss_image2


    ## シナプスのリソース量を考慮した出力値(x) と入力値（u_a1,u_a2,u_a3） ///////
    def adaptive_synapse(x, u):
        v = u * x
        for _ in range(substep):
            u = u + (-(tau * u + rec) * x + rec * (1.0 - u)) / substep
        return v, u


    def ImgConvert(img):
        convert_img = np.clip(img * 255, a_min=0, a_max=255).astype(np.uint8)
        return convert_img

    hc = LIFNeuron(N=n_bins, poisson=poisson)
    a1 = LIFNeuron(N=n_a1, adapt=True, lateral=lateral) # WTA機構のお話はしている（即抑制→発火したら周りを抑制していくよ　dogfilterでやっている）
    a2 = LIFNeuron(N=n_a2, adapt=True, lateral=lateral)
    a3 = LIFNeuron(N=n_a3, adapt=True, lateral=lateral)

    u_hc = np.ones(n_bins, dtype=np.float32)
    s_hc = np.zeros(n_bins, dtype=np.float32)

    u_a1 = np.zeros(n_a1, dtype=np.float32)
    s_a1 = np.zeros(n_a1, dtype=np.float32)
    w_a1 = np.random.randn(n_bins, n_a1) * 0.1 + 0.5

    u_a2 = np.zeros(n_a2, dtype=np.float32)
    s_a2 = np.zeros(n_a2, dtype=np.float32)
    w_a2 = np.random.randn(n_a1, n_a2) * 0.1 + 0.5

    u_a3 = np.zeros(n_a3, dtype=np.float32)
    s_a3 = np.zeros(n_a3, dtype=np.float32)
    w_a3 = np.random.randn(n_a2, n_a3) * 0.1 + 0.5

    # データの読み込み
    if load:
        w_a1 = np.load(log_path + 'w_a1.npy')
        w_a2 = np.load(log_path + 'w_a2.npy')
        w_a3 = np.load(log_path + 'w_a3.npy')

    log_a1_list = []
    log_a2_list = []
    log_a3_list = []

    log_w = []

    file_list = []
    for f in os.listdir(dataset_path):  # 指定されたディレクトリ内のファイルをループ
        if f.endswith('.wav'):  # ．wav拡張子のファイルを検索
            fpath = dataset_path + f  # fpathを生成しリストに追加
            file_list.append(fpath)

    file_list = np.array(file_list)

    while True:
        # 入力をcqt変換
        np.random.shuffle(file_list)
        for idx, fpath in enumerate(file_list):  # file_listの各要素に対してインデックス番号とファイルパスを取得
            wav, sr = librosa.load(path=fpath, sr=sr)
            csp = librosa.cqt(wav, sr=sr, hop_length=hop_length, fmin=librosa.note_to_hz('C1'),
                              bins_per_octave=bins_per_octave, n_bins=n_bins)

            cspr = csp.real  # 複素数の実部を取得
            lcspr = np.log(cspr.clip(0, None) + 1)
            # print(f'lcspr: {lcspr.shape}')
            log_hc = np.zeros_like(cspr)  # csprの形状の配列を０で初期化 2次元
            # print(f'cspr: {cspr.shape}')
            log_a1 = np.zeros((n_a1, cspr.shape[1]))  # axis=0にn_a1個、axis=1にcspr.shape[1]この2次元配列を０で初期化
            # print(f'log_a1: {log_a1.shape}')
            log_a2 = np.zeros((n_a2, cspr.shape[1]))
            # print(f'log_a2: {log_a2.shape}')
            log_a3 = np.zeros((n_a3, cspr.shape[1]))

            plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.amplitude_to_db(cspr, ref=np.max), bins_per_octave=bins_per_octave, sr=sr,
                                     x_axis='time', y_axis='cqt_note', cmap="jet")
            plt.title('Constant-Q power spectrum')
            plt.colorbar(format="%+2.0f dB")
            plt.savefig('cqt_spectrum.png')
            plt.close()

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
            dog_image = dog_filter(original_image)
            value_range = max(abs(dog_image.min()), abs(dog_image.max()))
            normalized_dog_image = np.clip(dog_image, 0, value_range)  # opencv用処理
            normalized_dog_image = (normalized_dog_image / value_range * 255).astype(np.uint8)  # opencv用処理
            # print(f'normalized_dog_image_max : {normalized_dog_image.max()}')

            # ニューラルネットワークの各層の更新
            for t in range(normalized_dog_image.shape[1]):
                # update input
                x = normalized_dog_image[:,t] / normalized_dog_image.max()  # 入力データの配列で、時刻tにおけるすべての入力を取得　データの最大値で割ることでxを正規化
                u_a1 = s_hc @ w_a1  # u_a1;ニューロン層a1への入力　（s＿hc；前層のスパイク活動）
                u_a2 = s_a1 @ w_a2
                u_a3 = s_a2 @ w_a3

                # update neuron
                c_hc, s_hc, v_hc = hc.run(x * u_hc)  # hc層はx*u_hcを使って更新　新しいシナプス状態（c_a1）、スパイク、膜電位を返す
                c_a1, s_a1, v_a1 = a1.run(u_a1)  # a1層はu_a1によって更新
                c_a2, s_a2, v_a2 = a2.run(u_a2)
                c_a3, s_a3, v_a3 = a3.run(u_a3)

                for _ in range(substep):  # ここの処理がわかりません
                    u_hc = u_hc - (u_hc + rgan) * (tgan) * x / substep + rgan * (1.0 - u_hc) / substep
                    u_hc = u_hc.clip(0, 1)

                # update weight STDPの更新方法を採用
                dw_a1 = stdp(s_hc, c_hc.clip(0, cmax - 1), s_a1, c_a1.clip(0, cmax - 1))
                dw_a2 = stdp(s_a1, c_a1.clip(0, cmax - 1), s_a2, c_a2.clip(0, cmax - 1))
                dw_a3 = stdp(s_a2, c_a2.clip(0, cmax - 1), s_a3, c_a3.clip(0, cmax - 1))

                w_a1 = (w_a1 + eta * (w_a1) * (1.0 - w_a1) * dw_a1).clip(0, 1)
                w_a2 = (w_a2 + eta * (w_a2) * (1.0 - w_a2) * dw_a2).clip(0, 1)
                w_a3 = (w_a3 + eta * (w_a3) * (1.0 - w_a3) * dw_a3).clip(0, 1)

                # log
                log_hc[:, t] = s_hc
                log_a1[:, t] = s_a1
                log_a2[:, t] = s_a2
                log_a3[:, t] = s_a3

                # print(f'w-a1 : {w_a1.shape}')
                # print(f'w-a2 : {w_a2.shape}')

                if t == lcspr.shape[1] - 1:
                    print("np_mean")
                    print(np.mean(log_a1, axis=1).shape)
                    log_a2_list.append(np.mean(log_a2, axis=1))
                    log_a3_list.append(np.mean(log_a3, axis=1))
                    epoch_count += 1
                    log_a1_list.append(np.mean(log_a1, axis=1))
                    print(f'counter:{epoch_count}')

                    if draw:
                        w_a2_img = (w_a1[:, :, None] * w_a2).mean(1)
                        w_a3_img = (w_a2[:, :, None] * w_a3).mean(1)
                        a1_simg = (log_a1.T[:, :, None] * w_a1.T).mean(axis=1)
                        a2_simg = (log_a2.T[:, :, None] * w_a2_img.T).mean(axis=1)
                        a3_simg = (log_a3.T[:, :, None] * w_a3_img.T).mean(axis=1)
                        line = np.ones(len(lcspr[1]))


                        cv2.imshow('w_a1', w_a1 / w_a1.max())  # ＳＴＤＰによって更新された一層目の重み（正規化）
                        cv2.imshow('w_a2', w_a2 / w_a2.max())
                        cv2.imshow('w_a3', w_a3 / w_a3.max())

                        cv2.imshow('csp_and_Dogfiltered_lcspr_img', np.vstack((lcspr / lcspr.max(), line, normalized_dog_image / 255)))  # スケールを合わせるための/255
                        cv2.imshow('log_a1_img', np.vstack((lcspr / lcspr.max(), line, normalized_dog_image / 255, line,
                                                            log_hc, line, a1_simg.T / a1_simg.max(), line,
                                                            a2_simg.T / a2_simg.max(), line,
                                                            a3_simg.T / a3_simg.max())))


                        print((w_a1 * (1 - w_a1)).mean(), (w_a2 * (1 - w_a2)).mean(), (w_a3 * (1 - w_a3)).mean())
                        log_w.append(
                            ((w_a1 * (1 - w_a1)).mean(), (w_a2 * (1 - w_a2)).mean(), (w_a3 * (1 - w_a3)).mean()))


                    # キーコマンド設定
                    key = cv2.waitKey(1)
                    if key == 27:
                        break

                    elif key == ord('d'):
                        draw = not draw

                if epoch_count == 10000:
                    cv2.destroyAllWindows()
                    exit()

# cv2.destroyAllWindows()








ulog = []
vlog = []

x = np.zeros(N)
u = np.zeros(N)
signal = True

uimg = np.zeros((N, T))
vimg = np.zeros((N, T))
for t in range(T):
    if t % 50 == 0:
        signal = not signal

    x[:] = signal * np.arange(N) / (N - 1)
    v, u = adaptive_synapse(x, u)

    uimg[:, t] = u
    vimg[:, t] = v

    cv2.imshow('log', np.vstack((uimg, vimg)))
    cv2.waitKey(1)

plt.plot(uimg[-1])
plt.plot(vimg[-1])
plt.show()