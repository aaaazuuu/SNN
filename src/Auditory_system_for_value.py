import datetime
import os

import numpy as np
import librosa
import librosa.display
import cv2
import matplotlib.pyplot as plt
import scipy
from numpy.core.shape_base import vstack
from numpy.lib.format import write_array_header_1_0

from numpy.ma.core import array
from numpy.lib.twodim_base import flipud, fliplr

# 構築した関数ファイルを持ってくる
import utils.dog_filter_utils
import utils.cos_similarity_utils
import utils.parameter

#ここを学習したｗのlogのpathに変更する
import got_studied_w_path_list.studied_only_piano_path


from lib.snn import LIFNeuron

if __name__ == '__main__':
    # 今の日付・時刻を取得（datetime.datetime.now()），文字列に変換する（.strftime("%Y_%m%d_%H%M")）
    now = datetime.datetime.now().strftime("%Y_%m%d_%H%M")

    # それぞれの変数名で、‘ログを保存するためのディレクトリパス’，‘図やグラフを保存する為のディレクトリパス’，‘データセットのパス’を指定する
    log_evaluation_path = f'../log_evaluation/{now}/'
    fig_evaluation_path = f'../fig_evaluation/{now}/'
    dataset_for_evaluation_path = '../dataset_for_evaluation/bright_piano/' # 評価用のデータセットパスに変更してください

    # パスが存在しなかった場合、log_eva_pathとfig_eva_pathを作成する（存在していれば何もない）
    os.makedirs(log_evaluation_path, exist_ok=True)
    os.makedirs(fig_evaluation_path, exist_ok=True)

    # 他のファイルで構築した関数を変数化
    dog_filter = utils.dog_filter_utils.dog_filter
    cossim_w_a1 = utils.cos_similarity_utils.CosSim_w_a1
    cossim_w_a2 = utils.cos_similarity_utils.CosSim_w_a2
    cossim_w_a3 = utils.cos_similarity_utils.CosSim_w_a3

    # 学習後のw（w_a1,w_a2,w_a3）を他のファイルから取得 →　ここを学習したｗのlogのpathに変更する
    w_a1 = got_studied_w_path_list.studied_only_piano_path.w_a1
    w_a2 = got_studied_w_path_list.studied_only_piano_path.w_a2
    w_a3 = got_studied_w_path_list.studied_only_piano_path.w_a3

    print(f'w_a1.shape{w_a1.shape}')
    print(f'w_a2.shape{w_a2.shape}')
    print(f'w_a3.shape{w_a3.shape}')


    # 各パラメータ設定値を他のファイルから取得
    epoch_count = utils.parameter.epoch_count
    n_fft = utils.parameter.n_fft
    sr = utils.parameter.sr
    poisson = utils.parameter.poisson
    n_bins = utils.parameter.n_bins
    hop_length = utils.parameter.hop_length
    bins_per_octave = utils.parameter.bins_per_octave
    lateral = utils.parameter.lateral
    load = utils.parameter.load
    draw = utils.parameter.draw

    exc_tau = utils.parameter.exc_tau
    exc_scale = utils.parameter.exc_scale

    inh_tau = utils.parameter.inh_tau
    inh_scale = utils.parameter.inh_scale

    n_a1 = utils.parameter.n_a1
    n_a2 = utils.parameter.n_a2
    n_a3 = utils.parameter.n_a3

    # decay 1:on / 0:off
    cell_type = utils.parameter.cell_type
    substep = utils.parameter.substep

    tgan = utils.parameter.tgan
    rgan = utils.parameter.rgan

    if cell_type == 1:
        tgan = 1.0
    rgan = 0.05


    def ImgConvert(img):
        convert_img = np.clip(img * 255, a_min=0, a_max=255).astype(np.uint8)
        return convert_img


    hc = LIFNeuron(N=n_bins, poisson=poisson)
    a1 = LIFNeuron(N=n_a1, adapt=True, lateral=lateral)  # WTA機構のお話はしている（即抑制→発火したら周りを抑制していくよ　dogfilterでやっている）
    a2 = LIFNeuron(N=n_a2, adapt=True, lateral=lateral)
    a3 = LIFNeuron(N=n_a3, adapt=True, lateral=lateral)

    u_hc = np.ones(n_bins, dtype=np.float32)
    s_hc = np.zeros(n_bins, dtype=np.float32)

    u_a1 = np.zeros(n_a1, dtype=np.float32)
    s_a1 = np.zeros(n_a1, dtype=np.float32)

    u_a2 = np.zeros(n_a2, dtype=np.float32)
    s_a2 = np.zeros(n_a2, dtype=np.float32)

    u_a3 = np.zeros(n_a3, dtype=np.float32)
    s_a3 = np.zeros(n_a3, dtype=np.float32)


    # スパイクの時間方向に平均を取った数値を入れる為の空リスト
    log_a1_list = []
    log_a2_list = []
    log_a3_list = []

    log_w = []

    file_list = []
    for f in os.listdir(dataset_for_evaluation_path):  # 指定されたディレクトリ内のファイル名をリストとして取得して、ファイルをループ
        if f.endswith('.wav'):  # ．wav拡張子のファイルを検索
            fpath = dataset_for_evaluation_path + f  # ファイル名をフルパスに変換してリストに追加
            file_list.append(fpath)

    file_list = np.array(file_list)

    while True:
        # 入力をcqt変換
        # np.random.shuffle(file_list)
        for idx, fpath in enumerate(file_list):  # file_listの各要素に対してインデックス番号とファイルパスを取得
            if idx == len(file_list)-1:
                np.save(log_evaluation_path + 'w_a1.npy', w_a1)
                np.save(log_evaluation_path + 'w_a2.npy', w_a2)
                np.save(log_evaluation_path + 'w_a3.npy', w_a3)
                np.save(log_evaluation_path + 'w_a1_cossim.npy', cossim_w_a1()[2])
                np.save(log_evaluation_path + 'w_a2_cossim.npy', cossim_w_a2()[2])
                np.save(log_evaluation_path + 'w_a3_cossim.npy', cossim_w_a3()[2])
                np.save(log_evaluation_path + 'log_w.npy', np.array(log_w))
                np.savetxt(log_evaluation_path + 'log_w.csv', np.array(log_w), delimiter=',')

            # スパイクのindex=１の平均値を取得して配列に格納
            # 入力データの初めと最後だけ記録する
            if 0 <= epoch_count <= len(file_list)-1:
                print('save_log')
                log_a1_save = np.array(log_a1_list)
                print(f'log_a1_save.shape: {log_a1_save.shape}')
                log_a2_save = np.array(log_a2_list)
                log_a3_save = np.array(log_a3_list)

                np.save(log_evaluation_path + 'log_a1_mean_index_1.npy', log_a1_save)
                np.save(log_evaluation_path + 'log_a2_mean_index_1.npy', log_a2_save)
                np.save(log_evaluation_path + 'log_a3_mean_index_1.npy', log_a3_save)

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
                x = normalized_dog_image[:,
                    t] / normalized_dog_image.max()  # 入力データの配列で、時刻tにおけるすべての入力を取得　データの最大値で割ることでxを正規化
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

                # log
                log_hc[:, t] = s_hc
                log_a1[:, t] = s_a1
                log_a2[:, t] = s_a2
                log_a3[:, t] = s_a3

                # print(f'w-a1 : {w_a1.shape}')
                # print(f'w-a2 : {w_a2.shape}')

                # 画像描画
                save_interval = len(file_list)-1  # 何ステップ目で保存するかのパラメータ

                if t == lcspr.shape[1] - 1:
                    print("np_mean")
                    print(np.mean(log_a1, axis=1).shape)
                    log_a2_list.append(np.mean(log_a2, axis=1))
                    log_a3_list.append(np.mean(log_a3, axis=1))
                    epoch_count += 1
                    log_a1_list.append(np.mean(log_a1, axis=1))
                    print(f'counter:{epoch_count+1}')

                    if draw:
                        w_a2_img = (w_a1[:, :, None] * w_a2).mean(1)
                        w_a3_img = (w_a2[:, :, None] * w_a3).mean(1)
                        a1_simg = (log_a1.T[:, :, None] * w_a1.T).mean(axis=1)
                        a2_simg = (log_a2.T[:, :, None] * w_a2_img.T).mean(axis=1)
                        a3_simg = (log_a3.T[:, :, None] * w_a3_img.T).mean(axis=1)
                        line = np.ones(len(lcspr[1]))

                        cv2.imshow('CosSim_w_a1', cossim_w_a1()[0])  # w_a1のcos類似度
                        cv2.imshow('CosSim_w_a2', cossim_w_a2()[0])  # w_a2のcos類似度
                        cv2.imshow('CosSim_w_a3', cossim_w_a3()[0])  # w_a3のcos類似度
                        cv2.imshow('Dogfiltered_lcspr_img', normalized_dog_image)  # DOGフィルターをかけた後のlcspr画像（側方抑制機構）
                        cv2.imshow('log_a1', log_a1)  # スパイクの活動状態を時間ステップ分スタック
                        cv2.imshow('log_a2', log_a2)
                        cv2.imshow('log_a3', log_a3)
                        cv2.imshow('w_a1', w_a1 / w_a1.max())  # ＳＴＤＰによって更新された一層目の重み（正規化）
                        cv2.imshow('w_a2', w_a2 / w_a2.max())
                        cv2.imshow('w_a3', w_a3 / w_a3.max())
                        cv2.imshow('w_a2_img', w_a2_img / w_a2_img.max())
                        cv2.imshow('w_a3_img', w_a3_img / w_a3_img.max())
                        cv2.imshow('csp_and_Dogfiltered_lcspr_img', np.vstack(
                            (lcspr / lcspr.max(), line, normalized_dog_image / 255)))  # スケールを合わせるための/255
                        cv2.imshow('log_a1_img', np.vstack((lcspr / lcspr.max(), line, normalized_dog_image / 255, line,
                                                            log_hc, line, a1_simg.T / a1_simg.max(), line,
                                                            a2_simg.T / a2_simg.max(), line,
                                                            a3_simg.T / a3_simg.max())))
                        cossim_1, mean_cossim_1, s1, t1 = cossim_w_a1()
                        cossim_2, mean_cossim_2, s2 = cossim_w_a2()
                        cossim_3, mean_cossim_3, s3 = cossim_w_a3()
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
                        log_w.append(
                            ((w_a1 * (1 - w_a1)).mean(), (w_a2 * (1 - w_a2)).mean(), (w_a3 * (1 - w_a3)).mean()))

                        # log_w.csvに書き込む
                        save_path = os.path.join(log_evaluation_path, "log_w.csv")  # pathの場所を参照する
                        np.savetxt(save_path, np.array(log_w).T, delimiter=",")

                    if epoch_count % save_interval == 0:
                        print(f'counter-{epoch_count+1}:save')
                        now_2 = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
                        cv2.imwrite(fig_evaluation_path + 'CosSim_w_a1-' + str(epoch_count+1) + 'time_' + now_2 + '.png',
                                    ImgConvert(cossim_w_a1()[0]))

                        cv2.imwrite(fig_evaluation_path + 'CosSim_w_a2-' + str(epoch_count+1) + 'time_' + now_2 + '.png',
                                    ImgConvert(cossim_w_a2()[0]))

                        cv2.imwrite(fig_evaluation_path + 'CosSim_w_a3-' + str(epoch_count+1) + 'time_' + now_2 + '.png',
                                    ImgConvert(cossim_w_a3()[0]))

                        cv2.imwrite(fig_evaluation_path + 'log_a1_' + str(epoch_count+1) + 'time_' + now_2 + '.png',
                                    ImgConvert(log_a1))
                        cv2.imwrite(fig_evaluation_path + 'log_a2_' + str(epoch_count+1) + 'time_' + now_2 + '.png',
                                    ImgConvert(log_a2))
                        cv2.imwrite(fig_evaluation_path + 'log_a3_' + str(epoch_count+1) + 'time_' + now_2 + '.png',
                                    ImgConvert(log_a3))
                        cv2.imwrite(fig_evaluation_path + 'w_a1_' + str(epoch_count+1) + 'time_' + now_2 + '.png',
                                    ImgConvert(w_a1 / w_a1.max()))
                        cv2.imwrite(fig_evaluation_path + 'w_a2_' + str(epoch_count+1) + 'time_' + now_2 + '.png',
                                    ImgConvert(w_a2 / w_a2.max()))
                        cv2.imwrite(fig_evaluation_path + 'w_a3_' + str(epoch_count+1) + 'time_' + now_2 + '.png',
                                    ImgConvert(w_a3 / w_a3.max()))
                        cv2.imwrite(fig_evaluation_path + 'w_a2_img-' + str(epoch_count+1) + 'time_' + now_2 + '.png',
                                    ImgConvert(w_a2_img / w_a2_img.max()))
                        cv2.imwrite(fig_evaluation_path + 'w_a3_img-' + str(epoch_count+1) + 'time_' + now_2 + '.png',
                                    ImgConvert(w_a3_img / w_a3_img.max()))
                        cv2.imwrite(
                            fig_evaluation_path + 'csp_and_Dogfiltered_lcspr_img-' + str(epoch_count+1) + 'time_' + now_2 + '.png',
                            ImgConvert(np.vstack((lcspr / lcspr.max(), line, normalized_dog_image / 255))))
                        cv2.imwrite(fig_evaluation_path + 'log_a1_img-' + str(epoch_count+1) + 'time_' + now_2 + '.png', ImgConvert(
                            np.vstack((lcspr / lcspr.max(), line, log_hc, line, a1_simg.T / a1_simg.max(), line,
                                       a2_simg.T / a2_simg.max(), line, a3_simg.T / a3_simg.max()))))

                    # キーコマンド設定
                    key = cv2.waitKey(1)
                    if key == 27:
                        break

                    elif key == ord('d'):
                        draw = not draw

                if epoch_count == len(file_list)+1:
                    cv2.destroyAllWindows()
                    exit()

    # cv2.destroyAllWindows()