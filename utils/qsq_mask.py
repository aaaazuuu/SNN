import numpy as np
import cv2
import librosa

n_bins = 256
bins_per_octave = 12 * 3 # 1オクターブを36周波数分に分ける

def qsq_mask():  # このフィルタ解析するときどこかのタイミングでかけたい

    n_oct = 128  # n_oct倍音まで考慮する
    ref = np.arange(1, n_oct) * librosa.note_to_hz('C2')  # ドの倍音128個分が出力される

    cqt_freqs = librosa.cqt_frequencies(n_bins=n_bins, bins_per_octave=bins_per_octave,
                                        fmin=librosa.note_to_hz('C2'))

    dif = (ref.reshape(-1, 1) - cqt_freqs.reshape(1, -1)) ** 2
    difmax = dif.argmin(axis=1).max()
    print('difmax', dif.argmin(axis=1).max())

    mask = np.zeros(difmax + 1)
    mask[dif.argmin(axis=1)] = 1.0

    print(mask.argmin(axis=0).max())

    return cv2.resize(mask[::-1], None, fx=10, fy=1)


mask = qsq_mask()
image = (mask * 255).astype(np.uint8)

cv2.imshow('qsq_mask', mask)  # C1の20個分の倍音と基音

save_path = "C:/workspace/fixed_SNN/utils/figs/base_C2_qsq_mask.jpg"
cv2.imwrite(save_path, image)

cv2.waitKey(0)

