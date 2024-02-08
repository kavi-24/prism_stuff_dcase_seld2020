import numpy as np
import matplotlib.pyplot as plt


def read_npy(path: str):
    return np.load(path)


Z = read_npy(
    r"DCASE2020_SELD_dataset\feat_label\mic_dev\fold1_room1_mix001_ov1.npy")

""" plt.imshow(np.transpose(Z), cmap='jet', origin='lower', aspect='auto')
plt.colorbar()
plt.show() """

print(Z.shape)
plt.specgram(Z[0])
plt.show()

[
    [float, float, ..., 640],
    ...  # 3000
]
