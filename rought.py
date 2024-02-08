import sys
import numpy as np

# print(sys.argv)
foldroom1mix001ov1 = np.load(r"DCASE2020_SELD_dataset\feat_label\mic_dev\fold1_room1_mix001_ov1.npy")
print(len(foldroom1mix001ov1[0]))