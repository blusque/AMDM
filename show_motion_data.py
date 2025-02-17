from dataset.util.bvh import read_bvh_loco

def read_bvh_file(bvh_file):
    bvh = read_bvh_loco(bvh_file, 1, 120, 180)
    return bvh

import json
import numpy as np

with open('/home/blusque/AMDM/data/MotionVivid/dataset_info.json') as f:
    data = json.load(f)
    mean = np.array(data['mean'])
    std = np.array(data['std'])

bvh_file = './data/MotionVivid/s004_angry_fw.bvh'
bvh, _ = read_bvh_file(bvh_file)
print(bvh.shape)
bvh = (bvh - mean) / std
key_points = {
    0: 'Global',
    3: "Positions",
    3 + 3 * 24: "Velocities",
    3 + 3 * 24 * 2: "Rotations",
}
data = ''
for i in range(bvh.shape[1]):
    if i != 0 and i % 3 == 0:
        print(data)
        data = ''
    if i in key_points:
        print(key_points[i])
    data += str(bvh[2][i]) + ' '
print(data)
# print(bvh[2])

from scipy.spatial.transform import Rotation as R
import numpy as np

axis = np.array([1, 0, 0])
alpha = 10
rot = R.from_rotvec(axis * alpha, degrees=True)
print(rot.as_quat())