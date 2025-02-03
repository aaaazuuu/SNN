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
    dataset_path = '../dataset/bright_piano/'

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


    def im2col(input_data, filter_h, filter_w, stride=1, pad=0): # 畳み込み用に画像データを2次元配列に変換する操作
        N, C, H, W = input_data.shape  # 入力データの形状　N；バッチサイズ、ｃ；チャネル数、Ｈ；高さ、Ｗ；幅
        filter_h, filter_w, _, _ = w.shape  # フィルタの形状を取得　filter_h;フィルターの高さ、filter_w;フィルターの幅
        out_h = (H + 2 * pad - filter_h) // stride + 1  # 出力の後の画像サイズの高さ
        out_w = (W + 2 * pad - filter_w) // stride + 1  #　出力の幅　計算方法　公式
        img = np.pad(input_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)  # 入力データにパディングを追加　ゼロパディング
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)) # 出力用の配列を初期化　6次元

        for y in range(filter_h):  # 入力データのスライドウィンドウの作成　入力データimgに対してフィルターをスライドさせながら対応するcol配列に格納する どんな処理してるのー？
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                col[:, :, y, x] = img[:, :, y:y_max:stride, x:x_max:stride]
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1) # col配列を転置してから理シェイプする　より、col配列は２次元の行列になり各行がフィルターの適用される領域を示す
        #                   N  H, W, C  Fh Fw         N*H*W x C*Fh*Fw

        # print(col.shape)
        return col


    def conv2d(x, w, stride=1, pad=0):  # 2次元畳み込みを行う関数
        N, C, H, W = x.shape
        Fh, Fw, Fc, Fk = w.shape  # フィルターの形状取得　fc；入力チャネル数　fｋ；出力チャネル数
        out_h = (H + 2 * pad - Fh) // stride + 1  # 出力の高さ
        out_w = (W + 2 * pad - Fw) // stride + 1  # 出力の幅
        xcol = im2col(x, Fh, Fw, stride, pad)  # im2col関数を使って２次元配列xcolに変換 フィルターが適用される各領域を１行に持つ行列

        # x [H*W,C*Fh*Fw]

        wcol = w.transpose(2, 0, 1, 3).reshape(Fc * Fh * Fw, Fk)  # フィルターのすべての要素を１列に持つ行列

        # w [Fh*Fw*C,K]

        out = np.dot(xcol, wcol)  # 行列の積の計算
        out = out.reshape(1, out_h, out_w, Fk)

        # 最終的な出力の形状
        # out [N, Fk, out_h, out_w]
        #                    N,C,H,W

        return out.transpose(0, 3, 1, 2)


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


    def stdp4conv2d(sin, cin, sout, cout, fsize, stride=1, pad=0):  # 岩舘先生に聞く 通常の全結合層のstdpとは違う
        sinshape = sin.shape

        sincol = im2col(sin, fsize, fsize, stride, pad)
        ltpin = ltp[cin.astype(np.int32)]
        ltpcol = im2col(ltpin, fsize, fsize, stride, pad)

        soutshape = sout.shape
        soutcol = sout.reshape(soutshape[0] * soutshape[1], soutshape[2] * soutshape[3])

        ltdcol = ltd[cout.astype(np.int32)].reshape(soutshape[0] * soutshape[1], soutshape[2] * soutshape[3])

        exc = np.dot(soutcol, ltpcol).reshape(soutshape[1], sinshape[1], fsize, fsize).transpose(2, 3, 1, 0) / (
                soutshape[2] * soutshape[3])
        inh = np.dot(ltdcol, sincol).reshape(soutshape[1], sinshape[1], fsize, fsize).transpose(2, 3, 1, 0) / (
                soutshape[2] * soutshape[3])

        # in = [N*H*W,C*Fh*Fw]
        # out = [N*K, H*W]
        # out * in = [K,C*Fh*Fw] -> [Fh,Fw,C,K]
        # w = [Fh,Fw,C,K]
        return exc - inh

    ## for dog_filter
    def _convolve2d(image, kernel):  # 入力画像に対して畳み込みフィルタ（kernel）を適用している
        shape = (image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1) + kernel.shape  # 出力shapeの決定
        strides = image.strides * 2  # メモリ上でのアクセスの調節 カーネルが画像内で移動しながら演算ができるようになる
        strided_image = np.lib.stride_tricks.as_strided(image, shape, strides)  ## 配列のstrideを調節して、元の配列のデータを再構築せずに異なる形で見ることができる　imageを新しいshapeとstridesに基づいて再構築
        return np.einsum('kl,ijkl->ij', kernel, strided_image)  ## karnelとstrided_imageの要素を適切にかけ合わせて畳み込みの結果を得ることができる


    ## for dog_filter
    def _convolve2d_multichannel(image, kernel):  # カラー画像の時に使う
        convolved_image = np.empty((image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1, image.shape[2]))  # image.shape[2]は、チャネル数RGB
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
        kernel = np.fromfunction(lambda y, x: np.exp(-((x - center[1]) ** 2 + (y - center[0]) ** 2) / sigma2), size)  # fromfunction(lambda 引数: 返り値, outputの行列の形,)
        kernel = kernel / np.sum(kernel)  # 重みの合計が１になるように正規化している
        return kernel

    # dog_filter
    def dog_filter(image, sigma1=2.0, sigma2=1.0, size=(7, 1), boundary='edge'):  # edge;境界付近でデータが不足した場合、最も近い端の値をコピーして計算に使用
        kernel1 = create_gaussian_kernel(size=size, sigma=sigma1)
        kernel2 = create_gaussian_kernel(size=size, sigma=sigma2)
        gauss_image1 = convolve2d(image, kernel1, boundary=boundary)
        gauss_image2 = convolve2d(image, kernel2, boundary=boundary)
        return gauss_image1 - gauss_image2

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

    # cos類似度計算
    def CosSim_w_a1():
        a = w_a1.copy().T
        b = w_a1.copy().T

        a_dis = np.sqrt((a ** 2).sum(axis=1))  # ユークリッド距離　l2norm →　割ると正規化される 過学習を防ぐためにも使えるから、勉強しておくと良い
        b_dis = np.sqrt((b ** 2).sum(axis=1))

        a = (a.T / (a_dis + 1e-8)).T  # ユークリッド距離　l2norm →　割ると正規化される
        b = (b.T / (b_dis + 1e-8)).T  # ユークリッド距離　l2norm →　割ると正規化される

        cossim_1 = np.matmul(a, b.T)
        # print(cossim_1.shape)

        t1 = np.triu(cossim_1, k=1)

        # 対角線を除いた平均類似度の計算
        s1 = cossim_1[np.triu_indices_from(cossim_1, k=1)]
        mean_cossim_1 = np.mean(s1)

        return cossim_1, mean_cossim_1, s1, t1

    def Correla_w_a1():
        x = np.corrcoef(w_a1.T)

        correla_w_a1 = x

        return correla_w_a1

    def CosSim_w_a2():
        a = w_a2.copy().T
        b = w_a2.copy().T

        a_dis = np.sqrt((a ** 2).sum(axis=1))  # ユークリッド距離　l2norm →　割ると正規化される 過学習を防ぐためにも使えるから、勉強しておくと良い
        b_dis = np.sqrt((b ** 2).sum(axis=1))

        a = (a.T / (a_dis + 1e-8)).T  # ユークリッド距離　l2norm →　割ると正規化される
        b = (b.T / (b_dis + 1e-8)).T  # ユークリッド距離　l2norm →　割ると正規化される

        cossim_2 = np.matmul(a, b.T)
        # print(cossim_2.shape)

        # 対角線を除いた平均類似度の計算
        s2 = cossim_2[np.triu_indices_from(cossim_2, k=1)]
        mean_cossim_2 = np.mean(s2)

        return cossim_2, mean_cossim_2, s2

    def Correla_w_a2():
        correla_w_a2 = np.corrcoef(w_a2.T)

        return correla_w_a2

    def CosSim_w_a3():
        a = w_a2.copy().T
        b = w_a2.copy().T

        a_dis = np.sqrt((a ** 2).sum(axis=1))  # ユークリッド距離　l2norm →　割ると正規化される 過学習を防ぐためにも使えるから、勉強しておくと良い
        b_dis = np.sqrt((b ** 2).sum(axis=1))

        a = (a.T / (a_dis + 1e-8)).T  # ユークリッド距離　l2norm →　割ると正規化される
        b = (b.T / (b_dis + 1e-8)).T  # ユークリッド距離　l2norm →　割ると正規化される

        cossim_3 = np.matmul(a, b.T)
        # print(cossim_2.shape)
        # 対角線を除いた平均類似度の計算
        s3 = cossim_3[np.triu_indices_from(cossim_3, k=1)]
        mean_cossim_3 = np.mean(s3)

        return cossim_3, mean_cossim_3, s3


    def Correla_w_a3():
        x = np.corrcoef(w_a3.T)
        correla_w_a3 = x
        return correla_w_a3

    # 一層目cos類似度のうち類似度の高い箇所のみのインデックスを取得→どこのニューロン番号のペアの相関が高いかを確認する用（どのようなパターンがそれぞれの音色に対応するかチェック）
    def take_cossim_index_w_a1():
        y = flipud(CosSim_w_a1()[3].T)  # cos_simの上三角行列を取得.T
        y2 = fliplr(y)
        h, v = np.where(y2 >= 0.3) # 0.4以上でtrue o.7以上の相関はなかった
        y3 = np.vstack((h, v))
        return y3

    def qsq_mask(): # このフィルタ解析するときどこかのタイミングでかけたい

        n_oct = 20  # n_oct倍音まで考慮する
        ref = np.arange(1, n_oct) * librosa.note_to_hz('C1')  # ドの倍音20個分が出力される

        cqt_freqs = librosa.cqt_frequencies(n_bins=n_bins, bins_per_octave=bins_per_octave,
                                            fmin=librosa.note_to_hz('C1'))

        dif = (ref.reshape(-1, 1) - cqt_freqs.reshape(1, -1)) ** 2
        difmax = dif.argmin(axis=1).max()
        print('difmax', dif.argmin(axis=1).max())

        mask = np.zeros(difmax + 1)
        mask[dif.argmin(axis=1)] = 1.0

        print(mask.argmin(axis=0).max())

        return cv2.resize(mask[::-1], None, fx=10, fy=1)
    mask = qsq_mask()

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
            if idx % 10 == 0:
                np.save(log_path + 'w_a1.npy', w_a1)
                np.save(log_path + 'w_a2.npy', w_a2)
                np.save(log_path + 'w_a3.npy', w_a3)
                np.save(log_path + 'w_a1_cossim.npy', CosSim_w_a1()[2])
                np.save(log_path + 'w_a2_cossim.npy', CosSim_w_a2()[2])
                np.save(log_path + 'w_a3_cossim.npy', CosSim_w_a3()[2])
                np.save(log_path + 'w_a1_cossim_num_check.npy', take_cossim_index_w_a1())
                np.save(log_path + 'log_w.npy', np.array(log_w))
                np.savetxt(log_path + 'log_w.csv', np.array(log_w), delimiter=',')


            # スパイクのindex=１の平均値を取得して配列に格納
            # 入力データが５２回転した頭から最後まで記録する　３８＊５２＝２０１４
            # if epoch_count == 2013:
            #     log_a1_list = []
            #     log_a2_list = []
            #     log_a3_list = []
            #
            # elif 2014 <= epoch_count <= 2051:
            #     print('save_log')
            #     log_a1_save = np.array(log_a1_list)
            #     print(log_a1_save.shape)
            #     log_a2_save = np.array(log_a2_list)
            #     log_a3_save = np.array(log_a3_list)
            #
            #     np.save(log_path + 'log_a1_mean_index_1.npy', log_a1_save)
            #     np.save(log_path + 'log_a2_mean_index_1.npy', log_a2_save)
            #     np.save(log_path + 'log_a3_mean_index_1.npy', log_a3_save)

            wav, sr = librosa.load(path=fpath, sr=sr)
            csp = librosa.cqt(wav, sr=sr, hop_length=hop_length, fmin=librosa.note_to_hz('C1'), bins_per_octave=bins_per_octave, n_bins=n_bins)

            cspr = csp.real  # 複素数の実部を取得
            lcspr = np.log(cspr.clip(0, None) + 1)
            # print(f'lcspr: {lcspr.shape}')
            log_hc = np.zeros_like(cspr)  # csprの形状の配列を０で初期化 2次元
            # print(f'cspr: {cspr.shape}')
            log_a1 = np.zeros((n_a1, cspr.shape[1])) # axis=0にn_a1個、axis=1にcspr.shape[1]この2次元配列を０で初期化
            # print(f'log_a1: {log_a1.shape}')
            log_a2 = np.zeros((n_a2, cspr.shape[1]))
            # print(f'log_a2: {log_a2.shape}')
            log_a3 = np.zeros((n_a3, cspr.shape[1]))



            plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.amplitude_to_db(cspr, ref=np.max), bins_per_octave=bins_per_octave, sr=sr, x_axis='time', y_axis='cqt_note', cmap="jet")
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
                original_image = original_image / np.iinfo(original_image.dtype).max # original_imageをclip(0,1)
            normalized_original_image = (original_image * 255).astype(np.uint8) # opencv用処理(lcspr画像と同じ)
            # print(f'original_image_max : {original_image.max()}')

            # dog_filtered_image
            dog_image = dog_filter(original_image)
            value_range = max(abs(dog_image.min()), abs(dog_image.max()))
            normalized_dog_image = np.clip(dog_image, 0, value_range) # opencv用処理
            normalized_dog_image = (normalized_dog_image / value_range * 255).astype(np.uint8) # opencv用処理
            # print(f'normalized_dog_image_max : {normalized_dog_image.max()}')

            # ニューラルネットワークの各層の更新
            for t in range(normalized_dog_image.shape[1]):
                # update input
                x = normalized_dog_image[:, t] / normalized_dog_image.max()  # 入力データの配列で、時刻tにおけるすべての入力を取得　データの最大値で割ることでxを正規化
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

                # 画像描画
                save_interval = 5000 # 何ステップ目で保存するかのパラメータ

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
                        a3_simg = (log_a3.T[:, :,  None] * w_a3_img.T).mean(axis=1)
                        line = np.ones(len(lcspr[1]))

                        cv2.imshow('CosSim_w_a1', CosSim_w_a1()[0])  # w_a1のcos類似度
                        cv2.imshow('Correla_w_a1', Correla_w_a1())  # w_a1の相関係数を用いた類似度
                        cv2.imshow('CosSim_w_a2', CosSim_w_a2()[0])  # w_a2のcos類似度
                        cv2.imshow('Correla_w_a2', Correla_w_a2())  # w_a2の相関係数を用いた類似度
                        cv2.imshow('CosSim_w_a3', CosSim_w_a3()[0])  # w_a3のcos類似度
                        cv2.imshow('Correla_w_a3', Correla_w_a3())  # w_a3の相関係数を用いた類似度
                        cv2.imshow('qsq_mask', mask)  # C1の20個分の倍音と基音
                        cv2.imshow('Dogfiltered_lcspr_img', normalized_dog_image)  # DOGフィルターをかけた後のlcspr画像（側方抑制機構）
                        cv2.imshow('log_a1', log_a1)  # スパイクの活動状態を時間ステップ分スタック
                        cv2.imshow('log_a2', log_a2)
                        cv2.imshow('log_a3', log_a3)
                        cv2.imshow('w_a1', w_a1 / w_a1.max())  # ＳＴＤＰによって更新された一層目の重み（正規化）
                        cv2.imshow('w_a2', w_a2 / w_a2.max())
                        cv2.imshow('w_a3', w_a3 / w_a3.max())
                        cv2.imshow('w_a2_img', w_a2_img / w_a2_img.max())
                        cv2.imshow('w_a3_img', w_a3_img / w_a3_img.max())
                        cv2.imshow('csp_and_Dogfiltered_lcspr_img', np.vstack((lcspr / lcspr.max(), line, normalized_dog_image/255))) # スケールを合わせるための/255
                        cv2.imshow('log_a1_img', np.vstack((lcspr / lcspr.max(), line, normalized_dog_image/255, line, log_hc, line,  a1_simg.T / a1_simg.max(), line, a2_simg.T / a2_simg.max(), line, a3_simg.T / a3_simg.max())))
                        cossim_1, mean_cossim_1, s1, t1 = CosSim_w_a1()
                        cossim_2, mean_cossim_2, s2 = CosSim_w_a2()
                        cossim_3, mean_cossim_3, s3 = CosSim_w_a3()
                        print(f'mean_cossim_1={mean_cossim_1}')
                        print(f'mean_cossim_2={mean_cossim_2}')
                        print(f'mean_cossim_3={mean_cossim_3}')
                        print(f'lcspr_shape: {lcspr.shape}')
                        print(f'w_a1: {w_a1.shape}')
                        print(f'w_a2: {w_a2.shape}')
                        print(f'w_a3: {w_a3.shape}')
                        print(f's_a1: {log_a1.shape}')
                        print(f's_a2: {log_a2.shape}')
                        print(f's_a3: {log_a3.shape}')
                        print(f'a1: {(a1_simg.T).shape}')
                        print(f'a2: {(a2_simg.T).shape}')
                        print(f'a3: {(a3_simg.T).shape}')

                        print((w_a1 * (1 - w_a1)).mean(), (w_a2 * (1 - w_a2)).mean(), (w_a3 * (1 - w_a3)).mean())
                        log_w.append(((w_a1 * (1 - w_a1)).mean(), (w_a2 * (1 - w_a2)).mean(), (w_a3 * (1 - w_a3)).mean()))

                        # log_w.csvに書き込む
                        save_path = os.path.join(log_path, "log_w.csv")  # pathの場所を参照する
                        np.savetxt(save_path, np.array(log_w).T, delimiter=",")


                    if epoch_count % save_interval == 0:
                        print(f'epoch{epoch_count}:save')
                        now_2 = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
                        cv2.imwrite(fig_path + 'CosSim_w_a1-' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(CosSim_w_a1()[0]))
                        cv2.imwrite(fig_path + 'Correlation_w_a1-' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(Correla_w_a1()))
                        cv2.imwrite(fig_path + 'CosSim_w_a2-' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(CosSim_w_a2()[0]))
                        cv2.imwrite(fig_path + 'Correlation_w_a2-' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(Correla_w_a2()))
                        cv2.imwrite(fig_path + 'CosSim_w_a3-' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(CosSim_w_a3()[0]))
                        cv2.imwrite(fig_path + 'Correlation_w_a3-' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(Correla_w_a3()))
                        cv2.imwrite(fig_path + 'log_a1_' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(log_a1))
                        cv2.imwrite(fig_path + 'log_a2_' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(log_a2))
                        cv2.imwrite(fig_path + 'log_a3_' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(log_a3))
                        cv2.imwrite(fig_path + 'w_a1_' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(w_a1 / w_a1.max()))
                        cv2.imwrite(fig_path + 'w_a2_' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(w_a2 / w_a2.max()))
                        cv2.imwrite(fig_path + 'w_a3_' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(w_a3 / w_a3.max()))
                        cv2.imwrite(fig_path + 'w_a2_img-' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(w_a2_img / w_a2_img.max()))
                        cv2.imwrite(fig_path + 'w_a3_img-' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(w_a3_img / w_a3_img.max()))
                        cv2.imwrite(fig_path + 'csp_and_Dogfiltered_lcspr_img-' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(np.vstack((lcspr / lcspr.max(), line, normalized_dog_image/255))))
                        cv2.imwrite(fig_path + 'log_a1_img-' + str(epoch_count) + 'time_' + now_2 + '.png', ImgConvert(np.vstack((lcspr / lcspr.max(), line, log_hc, line, a1_simg.T / a1_simg.max(), line, a2_simg.T / a2_simg.max(), line, a3_simg.T / a3_simg.max()))))

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
