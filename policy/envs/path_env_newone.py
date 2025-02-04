import policy.envs.base_env as base_env
from render.realtime.mocap_renderer import PBLMocapViewer
import dataset.util.geo as geo_util 

import os.path as osp
import copy
import torch
import numpy as np
import tkinter as tk
import gymnasium as gym
import policy.envs.target_env as target_env
from random import random

class PathEnv(target_env.TargetEnv):
    NAME = "PATH"
    def __init__(self, config, model, dataset, device):
        super().__init__(
            config, model, dataset, device
        )
        self.timestep = 0
        self.substep = 0

        # controller receives 4 upcoming targets
        self.lookahead = 4
        # time gap between each lookahead frame
        # 15 frames is 0.5 seconds in real-time
        self.lookahead_skip = 4
        self.lookahead_gap = 15
        self.expected_traj_length = self.lookahead * self.lookahead_gap + self.lookahead_skip
        self.big_err = 5
        self.force = torch.zeros(self.num_parallel, 2)
        condition_size = self.frame_dim * self.num_condition_frames
        self.observation_dim = condition_size + 2 * self.lookahead
        # self.observation_dim = condition_size + 2 * self.lookahead + 1

        high = np.inf * np.ones([self.observation_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        
        self.path, self.path_vel = self.sample_random_traj_nodr(1)
        self.change_path_rate = 0.3
        self.path = self.path[0]
        self.path_vel = self.path_vel[0]

        if self.is_rendered:
            self.path_offsets = torch.zeros(self.num_parallel, 1).long()
        else:
            self.path_offsets = torch.randint(
                0, self.path.size(0), (self.num_parallel, 1)
            ).long()

        self.facing_mode = self.sample_random_facing(1)[0]
        self.change_facing_gap = 12 * self.lookahead_gap
        
        # if self.is_rendered:
        #     #np.save(osp.join(self.int_output_dir,'traj.npy'), self.path.cpu().detach().numpy())
        #     self.viewer.add_path_markers(self.path)

    def add_force(self, force):
        if not isinstance(force, torch.Tensor):
            force = torch.tensor(force).float()
        self.force[:,] = force

    def sample_random_traj_nodr(self, num_parallel=1, std_acc=0.004, max_speed=0.03, eps=1e-8, vel=None, pos=None):
        if not self.is_rendered and self.max_timestep - self.timestep <= 0:
            return None, None
        traj_len = self.max_timestep - self.timestep if not self.is_rendered else self.expected_traj_length
        # traj_len = self.max_timestep - self.timestep
        traj = torch.zeros(num_parallel, traj_len, 2)
        traj_vel = torch.zeros(num_parallel, traj_len, 2)
        vel = vel if isinstance(vel, torch.Tensor) else torch.zeros(num_parallel,2)
        pos = pos if isinstance(pos, torch.Tensor) else torch.zeros(num_parallel,2)
        if len(vel.shape) == 1:
            vel = vel.unsqueeze(0)
            vel = vel.repeat(num_parallel,1)
        if len(pos.shape) == 1:
            pos = pos.unsqueeze(0)
            pos = pos.repeat(num_parallel,1)
        for i in range(traj_len):
            if self.is_rendered:
                # mat = self.get_rotation_matrix(-self.root_facing).clone().detach().cpu()
                # acc = torch.matmul(mat, self.force.unsqueeze(-1)).squeeze(-1)
                acc = self.force
            else:
                acc = torch.normal(mean=torch.zeros(num_parallel,2), std=torch.ones(num_parallel,2)*std_acc)
            # acc = torch.normal(mean=torch.zeros(num_parallel,2), std=torch.ones(num_parallel,2)*std_acc)
            vel += acc
            if torch.norm(acc) < 1e-8:
                vel = torch.normal(mean=torch.zeros(num_parallel,2), std=torch.ones(num_parallel,2)*0.0001)
            
            speed = torch.sqrt(torch.sum(vel*vel,dim=-1))
           
            speed_clamped = torch.clamp(speed, eps, max_speed)
            vel = vel/speed*speed_clamped
            
            pos += vel
            traj[:,i] = pos.clone()
            traj_vel[:,i] = vel.clone()
        traj = traj.to(self.device)
        traj_vel = traj_vel.to(self.device)
        return traj, traj_vel
    
    def sample_random_facing(self, num_parallel=1):
        facing = torch.randint(0, 2, (num_parallel, 1)).to(self.device)
        print(facing[0].item())
        return facing
    
    def randomize_path(self):
        index = self.timestep
        index = max(0, min(index, self.path.size(0)-1))
        # index = torch.clamp(index, 0, self.path.size(0)-1)
        new_path, new_path_vel = self.sample_random_traj_nodr(1, pos=self.path[0].clone().detach().cpu(),\
                                                              vel=self.path_vel[0].clone().detach().cpu())
        if new_path is None:
            return
        self.path[index:] = new_path.clone()
        self.path_vel[index:] = new_path_vel.clone()
        if self.is_rendered:
            self.viewer.add_path_markers(self.path)

    def update_path(self):
        self.path, self.path_vel = self.sample_random_traj_nodr(1, pos=self.path[1].clone().detach().cpu(),\
                                                              vel=self.path_vel[1].clone().detach().cpu())
        self.path = self.path[0]
        self.path_vel = self.path_vel[0]
        next_k = (
            torch.arange(0, self.lookahead) * self.lookahead_gap
            + self.lookahead_skip
        ) 
        next_k = torch.clamp(next_k, 0, self.path.size(0)-1)
        if self.is_rendered:
            self.viewer.add_path_markers(self.path[next_k])
    
    def given_traj(self, traj):
        self.path = traj
        if self.is_rendered:
            self.viewer.add_path_markers(self.path)

    def generate_amdm_traj(self, num_parallel=1):
        start_x_index = torch.randint(self.dataset.motion_flattened.shape[0],(1,))
        
        start_x = self.dataset.motion_flattened[start_x_index]
        start_x = torch.tensor(start_x, device = self.device).float()
        self.fixed_init_frame = start_x
        seqs =  self.model.eval_seq(start_x, None, self.max_timestep, num_parallel).cpu().detach().numpy()
        seqs = self.dataset.denorm_data(seqs)
        trajs = []
        for i in range(seqs.shape[0]):
            trajs.append(self.dataset.x_to_trajs(seqs[i]))
        trajs = torch.tensor(trajs).to(self.device).float()
        return trajs

    def reset(self, indices=None):
        if self.is_rendered:
            self.path_offsets = torch.zeros(self.num_parallel, 1).long()
        else:
            self.path_offsets = torch.randint(
                0, self.path.size(0), (self.num_parallel, 1)
            ).long()

        if indices is None:
            self.root_facing.fill_(0)
            self.root_xz.fill_(0)
            self.reward.fill_(0)
            self.timestep = 0
            self.substep = 0
            self.done.fill_(False)
            # value bigger than contact_threshold
            #self.foot_pos_history.fill_(1)

            self.reset_target()
            self.reset_initial_frames()
        else:
            self.root_facing.index_fill_(dim=0, index=indices, value=0)
            self.root_xz.index_fill_(dim=0, index=indices, value=0)
            self.reward.index_fill_(dim=0, index=indices, value=0)
            self.done.index_fill_(dim=0, index=indices, value=False)
           
            self.reset_target(indices)
            self.reset_initial_frames(indices)
            # value bigger than contact_threshold
            #self.foot_pos_history.index_fill_(dim=0, index=indices, value=1)
        
        self.path, self.path_vel = self.sample_random_traj_nodr(1)
        self.path = self.path[0]
        self.path_vel = self.path_vel[0]

        obs_components = self.get_observation_components()
        return torch.cat(obs_components, dim=1)
    
    def reset_initial_frames(self, frame_index=None):
        super().reset_initial_frames(frame_index)

        # set initial root position to random place on path
        #self.root_xz.copy_(self.path[self.path_offsets.squeeze(1)])
        self.root_xz.copy_(self.path[0])
        next_two = torch.arange(0, 2) * self.lookahead_gap + self.path_offsets + self.lookahead_skip
        
        next_two = torch.clamp(next_two, 0, self.path.size(0)-1)
        #delta = self.path[next_two[:, 1]] - self.path[next_two[:, 0]]
        facing = 0 #-torch.atan2(delta[:, 1], delta[:, 0]).unsqueeze(1)
        self.root_facing.copy_(facing)
        
        if self.is_rendered:
            # don't forget to convert feet to meters
            centre = self.path.mean(dim=0) * 0.3048
            xyz = torch.nn.functional.pad(centre, pad=[0, 1]).cpu().numpy()
            self.viewer.camera.lookat(xyz)

    def reset_target(self, indices=None, location=None):
        # don't add skip to accurate calculate is target is close
        # index = self.timestep + self.path_offsets.squeeze(1) + self.lookahead_skip
        # index = self.timestep % self.lookahead_gap + self.lookahead_skip
        index = self.lookahead_skip
        self.target.copy_(self.path[index])
        
        self.calc_potential()

        # move render target to get_delta_to_k_targets
        if self.is_rendered:
            self.viewer.update_target_markers(self.target)

    def get_delta_to_k_targets(self):
        # + lookahead_skip so it's not too close to character
        # next_k = (
        #     torch.arange(0, self.lookahead) * self.lookahead_gap
        #     + self.timestep
        #     + self.path_offsets
        #     + self.lookahead_skip
        # ) 
        next_k = (
            torch.arange(0, self.lookahead) * self.lookahead_gap
            + self.lookahead_skip
        ) 
        
        next_k = torch.clamp(next_k, 0, self.path.size(0)-1)

        # if self.is_rendered:
        #     self.viewer.update_target_markers(self.path[next_k])
        
        # (np x lookahead x 2) - (np x 1 x 2)
        target_delta = self.path[next_k] - self.root_xz.unsqueeze(1)
        # Should be negative because going from global to local
        mat = self.get_rotation_matrix(-self.root_facing)
        # (np x 1 x 2 x 2) x (np x lookahead x 1 x 2)
        delta = (mat.unsqueeze(1) * target_delta.unsqueeze(2)).sum(dim=-1)
        return delta

    def get_observation_components(self):
        deltas = self.get_delta_to_k_targets()
        condition = self.get_cond_frame()
        # facing = self.facing_mode.repeat(self.num_parallel, 1)
        # return condition, deltas.flatten(start_dim=1, end_dim=2), facing
        return condition, deltas.flatten(start_dim=1, end_dim=2)

    def dump_additional_render_data(self):
        return {
            "extra.csv": {"header": "Target.X, Target.Z", "data": self.target[0]},
            "root0.csv": {
                "header": "Root.X, Root.Z, RootFacing",
                "data": torch.cat((self.root_xz, self.root_facing), dim=-1)[0],
            },
        }
    
    def get_target_delta_and_angle(self):
        target_delta = self.target - self.root_xz
        target_angle = (
            torch.atan2(target_delta[:, 1], target_delta[:, 0]).unsqueeze(1)
            + self.root_facing + np.pi * self.facing_mode.float()
        )
        return target_delta, target_angle

    def calc_env_state(self, next_frame):
        self.next_frame = next_frame
        is_external_step = self.substep == 0

        if self.substep == self.frame_skip - 1:
            self.timestep += 1
        self.substep = (self.substep + 1) % self.frame_skip

        self.integrate_root_translation(next_frame)

        progress = self.calc_progress_reward()

        # Check if target is reached
        # Has to be done after new potentials are calculated
        target_dist = -self.linear_potential
        target_is_too_far = target_dist > self.big_err
        # dist_reward = target_is_too_far.float() * -2 * torch.exp(0.5 * target_dist) + \
        #     2 * torch.exp(0.5 * self.linear_potential) + progress
        dist_reward = 2 * torch.exp(0.5 * self.linear_potential) + progress

        if is_external_step:
            self.reward.copy_(dist_reward)
        else:
            self.reward.add_(dist_reward)

        target_is_close = target_dist < 0.2
        self.reward.add_(target_is_close.float() * (0.2 - target_dist) * 10.0)

        target_is_super_close = target_dist < 0.1
        self.reward.add_(target_is_super_close.float() * 20.0)

        # if self.is_rendered and self.timestep % self.lookahead_gap == 0:
        #     if random() < self.change_path_rate:
        #         self.randomize_path()
        # if self.is_rendered and self.timestep % self.lookahead_gap == 0:
        #     self.update_path()
        if self.is_rendered:
            self.update_path()
        # if self.timestep % self.change_facing_gap == 0:
        #     self.facing_mode = self.sample_random_facing(1)[0]
        # Need to reset target to next point in path
        # can only do this after progress is calculated
        self.reset_target()

        obs_components = list(self.get_observation_components())
        #obs_components[0] = obs_components[0].unsqueeze(0)
        #obs_components[1] = obs_components[1].unsqueeze(1)
        self.done.fill_(self.timestep >= self.max_timestep)
    
        # Everytime this function is called, should call render
        # otherwise the fps will be wrong
        self.render()

        return (
            torch.cat(obs_components, dim=-1),
            self.reward,
            self.done,
            {"reset": self.timestep >= self.max_timestep},
        )
    