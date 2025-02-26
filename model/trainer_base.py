import abc
import copy
import numpy as np

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

import util.vis_util as vis_util
import util.logging as logging_util
import util.save as save_util
import yaml

import os.path as osp

class BaseTrainer():
    def __init__(self, config, dataset, device):
        self.config = config
        self.device = device
        self.dataset = dataset

        optimizer_config = config['optimizer']
        self.batch_size = optimizer_config['mini_batch_size']
        self.num_rollout = optimizer_config['rollout']
        self.initial_lr = optimizer_config['initial_lr']
        self.final_lr = optimizer_config['final_lr']
        self.peak_student_rate = optimizer_config.get('peak_student_rate',1.0)
        self._get_schedule_samp_routines(config['optimizer'])
        
        test_config = config['test']
        self.test_interval = test_config["test_interval"]
        self.test_num_steps = test_config["test_num_steps"]
        self.test_num_trials = test_config["test_num_trials"]
        
        self.frame_dim = dataset.frame_dim
        self.train_dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.logger =  logging_util.wandbLogger(proj_name="{}_{}".format(self.NAME, dataset.NAME), run_name=self.NAME)

        self.plot_jnts_fn = self.dataset.plot_jnts if hasattr(self.dataset, 'plot_jnts') and callable(self.dataset.plot_jnts) \
                                                        else vis_util.vis_skel

        self.plot_traj_fn = self.dataset.plot_traj if hasattr(self.dataset, 'plot_traj') and callable(self.dataset.plot_traj) \
                                                        else vis_util.vis_traj
        return

    @abc.abstractmethod
    def train_loop(self, model):
        return    

    def _init_optimizer(self, model):
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.initial_lr)

    def _update_lr_schedule(self, optimizer, epoch):
        """Decreases the learning rate linearly"""
        lr = self.initial_lr - (self.initial_lr - self.final_lr) * epoch / float(self.total_epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _get_schedule_samp_routines(self, optimizer_config):
        self.anneal_times = optimizer_config['anneal_times']
        self.anneal_steps = optimizer_config['anneal_steps']
        self.initial_teacher_epochs = optimizer_config.get('initial_teacher_epochs',1)
        self.end_teacher_epochs = optimizer_config.get('end_teacher_epochs',1)
        self.teacher_epochs = optimizer_config['teacher_epochs']
        self.ramping_epochs = optimizer_config['ramping_epochs']
        self.student_epochs = optimizer_config['student_epochs']
        self.use_schedule_samp = self.ramping_epochs != 0 or self.student_epochs != 0
        
        self.initial_schedule = torch.zeros(self.initial_teacher_epochs)
        self.end_schedule = torch.zeros(self.end_teacher_epochs)
        self.sample_schedule = torch.cat([ 
                # First part is pure teacher forcing
                torch.zeros(self.teacher_epochs),
                # Second part with schedule sampling
                torch.linspace(0.0, self.peak_student_rate, self.ramping_epochs),
                # last part is pure student
                torch.ones(self.student_epochs) * self.peak_student_rate,

        ])
        self.sample_schedule = torch.cat([self.sample_schedule  for _ in range(self.anneal_times)], axis=-1)
        self.sample_schedule = torch.cat([self.initial_schedule, self.sample_schedule, self.end_schedule])
       
        self.total_epochs = self.sample_schedule.shape[0]


    def train_model(self, model, out_model_file, int_output_dir, log_file):
        self._init_optimizer(model)
        for ep in range(self.total_epochs):
            loss_stats = self.train_loop(ep, model)
            if ep == 0:
                continue
            if ep % self.test_interval == 0:
                num_nans = self.evaluate(ep, model, int_output_dir)
                save_util.save_weight(model, int_output_dir+'_ep{}.pth'.format(ep))
                save_util.save_weight(model, out_model_file)
                
            self.logger.log_epoch(loss_stats)
            self.logger.print_log(loss_stats)
            
        save_util.save_weight(model, out_model_file)

    def evaluate(self, ep, model, result_ouput_dir):
        model.eval()
        NaN_clip_num = 0

        for idx, (st_idx, ref_clip) in enumerate(zip(self.dataset.test_valid_idx, self.dataset.test_ref_clips)):
            print('Eval Index:',st_idx)
            test_out_lst = []
            test_local_out_lst = []
            extra_dict = {}

            start_x = torch.from_numpy(ref_clip[0]).float().to(self.device)
            
            if ep == 0:
                model_lst = self.dataset.data_component
                cur_jnts = []
                for mode in model_lst:
                    jnts_mode = self.dataset.x_to_jnts(self.dataset.denorm_data(ref_clip), mode=mode)
                    cur_jnts.append(jnts_mode)
                cur_jnts = np.array(cur_jnts)

                self.plot_jnts_fn(cur_jnts.squeeze(), result_ouput_dir+'/gt_{}'.format(st_idx))
                ref_clip = cur_jnts[[0],...]
            else:
                ref_clip = self.dataset.x_to_jnts(self.dataset.denorm_data(ref_clip), mode=self.dataset.data_component[0])[None,...]
            
            ref_local_clip = ref_clip - ref_clip[:,:,[0],:]

            test_out_lst.append(ref_clip.squeeze())
            test_data = model.eval_seq(start_x, extra_dict, self.test_num_steps, self.test_num_trials)
            test_data_long = model.eval_seq(start_x, extra_dict, 1000, 3)

            num_all = torch.numel(test_data)
            num_nans = torch.sum(torch.isnan(test_data))

            num_all_long = torch.numel(test_data_long)
            num_nans_long = torch.sum(torch.isnan(test_data_long))
        
            print('percent of nan frames : {}'.format(num_nans*1.0/num_all))
            print('percent of nan frames for long horizon gen : {}'.format(num_nans_long*1.0/num_all_long))
            should_plot = True
            if num_nans > 0:
                NaN_clip_num += 1
                should_plot = False
                #print('skip calc stats {} to save time'.format(st_idx))
                #if False:#NaN_clip_num >= len(self.dataset.test_valid_idx)-1:
                #continue # skip calc stats to save time
                        
            test_data = test_data.detach().cpu().numpy()

            for i in range(test_data.shape[0]):
                cur_denormed_test_data = self.dataset.denorm_data(copy.deepcopy(test_data[i]))
                cur_jnts = []
               
                for mode in self.dataset.data_component:
                    jnts_mode = self.dataset.x_to_jnts(cur_denormed_test_data, mode = mode)
                    cur_jnts.append(jnts_mode)

                    if mode == self.dataset.data_component[0]:
                        test_out_lst.append(jnts_mode)
                        jnts_mode_local = jnts_mode - jnts_mode[:,[0],:]  
                        test_local_out_lst.append(jnts_mode_local)
                cur_jnts = np.array(cur_jnts)
                if should_plot:
                    self.plot_jnts_fn(cur_jnts.squeeze(), result_ouput_dir+'/{}_{}'.format(st_idx,i))
            test_out_lst = np.array(test_out_lst)
            self.plot_traj_fn(test_out_lst, result_ouput_dir+'/{}'.format(st_idx))
            
            test_data_long = test_data_long.detach().cpu().numpy()
            test_out_long_lst = []
            for i in range(test_data_long.shape[0]):
                cur_denormed_test_data = self.dataset.denorm_data(copy.deepcopy(test_data_long[i]))
                cur_jnts = []
               
                for mode in self.dataset.data_component:
                    jnts_mode = self.dataset.x_to_jnts(cur_denormed_test_data, mode = mode)
                    cur_jnts.append(jnts_mode)

                    if mode == self.dataset.data_component[0]:
                        test_out_long_lst.append(jnts_mode)
                        jnts_mode_local = jnts_mode - jnts_mode[:,[0],:]  
                cur_jnts = np.array(cur_jnts)
                # if should_plot:
                #     self.plot_jnts_fn(cur_jnts.squeeze(), result_ouput_dir+'/{}_{}_long'.format(st_idx,i))
                # self.dataset.save_bvh(osp.join('/home/mjd/AMDM','out{}'.format(i)), cur_denormed_test_data)
            test_out_long_lst = np.array(test_out_long_lst)
            self.plot_traj_fn(test_out_long_lst, result_ouput_dir+'/{}_long'.format(st_idx))
        
        return NaN_clip_num
