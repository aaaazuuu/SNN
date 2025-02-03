# 各パラメータ設定値
epoch_count = -1
n_fft = 1024  # FFT（高速フーリエ変換）のサイズ
sr = 44100  # サンプリングレート　音声をデータにする時に、１秒間を何個の点に分割するかという意味　高いと滑らかな音の取得が可能
poisson = True
n_bins = 256
hop_length = 192
bins_per_octave = 12 * 3  # 1オクターブを36周波数分に分ける
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