import copy
import glob
import os
import csv
import numpy as np
import dataset.base_dataset as base_dataset
import dataset.util.bvh as bvh_util
import dataset.util.unit as unit_util
import dataset.util.geo as geo_util
import dataset.util.plot as plot_util
import os.path as osp

class MotionVivid(base_dataset.BaseMotionData):
    NAME = 'MOTIONVIVID'
    STYLE_MAP = {
        'neutral': 0,
        'angry': 1,
        'drunk': 2,
        'happy': 3,
        'fear': 4,
        'swimming': 5,
        'twofootjump': 6,
        'depressed': 7,
        'bigstep': 8
    }
    def __init__(self, config):
        self.only_forward = False
        if 'subject_name' in config['data']:
            self.subject_name = config['data']['subject_name']
            self.only_forward = config['data'].get('only_forward', False)
        super().__init__(config)
        
    def get_motion_fpaths(self):
        path =  osp.join(self.path,'{}_*.{}'.format(self.subject_name, 'bvh'))
        file_lst = glob.glob(path, recursive = True)
        if self.only_forward:
            file_lst = [f for f in file_lst if 'fr' in f or 'fw' in f.split('/')[-1]]
        return file_lst
    
    def process_data(self, fname):
        final_x, motion_struct = bvh_util.read_bvh_loco(fname, self.unit, self.fps, self.root_rot_offset)
        if self.data_trim_begin:
            final_x = final_x[self.data_trim_begin:]
        if self.data_trim_end:
            final_x = final_x[:self.data_trim_end]
        return final_x, motion_struct
    
    def process_label(self, fname):
        style = fname.split('_')[1]
        style_idx = self.STYLE_MAP[style]
        style_list = []
        for i in range(self.cur_length):
            style_list.append(style_idx)
        return style_list

    def load_new_data(self, path):
        x = self.process_data(path)
        x_normed = self.norm_data(x[0])
        #x_normed = self.transform_new_data(x_normed)
        return x_normed
    
    def plot_jnts(self, x, path=None):
        return plot_util.plot_multiple(x, self.links, plot_util.plot_lafan1, self.fps, path)
    
    def plot_traj(self, x, path=None):
        return plot_util.plot_traj_lafan1(x, path)
    
    def __getitem__(self, idx):
        idx_ = self.valid_idx[idx]
        motion = self.motion_flattened[idx_:idx_ + self.rollout]
        style = self.labels[idx_]
        return motion, style


