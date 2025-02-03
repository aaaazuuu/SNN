## まだ聴覚プログラムに組み込めていない

import numpy as np
import cv2
import matplotlib.pyplot as plt

substep = 2
tau = 0.0
rec = 0.5


def adaptive_synapse(x, u):
    v = u * x
    for _ in range(substep):
        u = u + (-(tau * u + rec) * x + rec * (1.0 - u)) / substep
    return v, u


ulog = []
vlog = []

N = 200
T = 500
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