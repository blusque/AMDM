
import glob

import os.path as osp
import os

translate = [0, 0, 500]

path = '/home/sy/repo/AMDM-public/output/inpaint/amdm_style100_consist_new_rollout_4/out_angle_0.bvh'
out_path = '/home/sy/repo/AMDM-public/output/inpaint/amdm_style100_consist_new_rollout_4/out_angle_0_tranlate.bvh'

#os.makedirs(out_path,exist_ok=True)
bvh_f = open(path,'r')
lines = bvh_f.readlines()
pre = []
records = []
count = 0
num_frames = 0
frame_ratr = 0
pre_flag = True

for i, line in enumerate(lines):
    if line.strip() == 'MOTION':
        pre_flag = False
    elif line.strip().split()[0] == "Frames:":
        num_frames = line.strip().split()[-1]
    elif line.strip().split()[0] == "Frame":
        frame_ratr = line.strip().split()[-1]
    else:   
        if pre_flag:
            pre.append(line)
        else:
            record_line = line
            root_pos = [str(eval(x)+translate[i]) for i, x in enumerate(record_line.split()[:3])]
            record_line = ' '.join(root_pos + record_line.split()[3:]) +'\n'
            records.append(record_line)
            count += 1
bvh_f.close()

out_name = out_path#osp.join(out_path,'translate.bvh')

with open(out_name, 'w') as f:
    for pre_line in pre:
        f.write(pre_line)
    f.write('MOTION\n')
    f.write('Frames: {}\n'.format(num_frames))
    f.write('Frame Time: {}\n'.format(frame_ratr))
    for record_line in records:
        f.write(record_line)