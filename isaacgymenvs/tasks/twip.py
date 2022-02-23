# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class Twip(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):

        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        #self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        #self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.control_mode = self.cfg["env"]["controlMode"]
        if self.control_mode == "velocity":
            self.max_velocity = self.cfg["env"]["maxVelocity"]
        if self.control_mode == "effort":
            self.max_effort = self.cfg["env"]["maxEffort"]
        self.gear_ratio = self.cfg["env"]["gearRatio"]
        #self.heading_weight = self.cfg["env"]["headingWeight"]
        #self.up_weight = self.cfg["env"]["upWeight"]
        #self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        #self.energy_cost_scale = self.cfg["env"]["energyCost"]
        #self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        #self.death_cost = self.cfg["env"]["deathCost"]
        #self.termination_height = self.cfg["env"]["terminationHeight"]
        self.reset_dist = self.cfg["env"]["resetDist"]
        self.free_dofs = self.cfg["env"]["freeDofs"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.cfg["env"]["numObservations"] = 7
        self.cfg["env"]["numActions"] = 2

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.body_state = gymtorch.wrap_tensor(body_state_tensor)
        self.body_ori = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self.body_linvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[...,7:10]
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()

        self.dt = self.cfg["sim"]["dt"]

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        assert "asset" in self.cfg["env"]
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),asset_root)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        if self.control_mode == "velocity":
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
        if self.control_mode == "effort":
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        #asset_options.angular_damping = 0.0

        ant_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # = self.gym.get_asset_actuator_properties(ant_asset)        
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.1, self.up_axis_idx))

        #self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.imu_frame_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)
        body_names = [self.gym.get_asset_rigid_body_name(ant_asset, i) for i in range(self.num_bodies)]
        self.num_dof = self.gym.get_asset_dof_count(ant_asset)
        dof_names = [self.gym.get_asset_dof_name(ant_asset, i) for i in range(self.num_dof)]
        # extremity_names = [s for s in body_names if "foot" in s]
        # self.extremities_index = torch.zeros(len(extremity_names), dtype=torch.long, device=self.device)
        print("Bodies: ", body_names)
        print("Dofs: ", dof_names)
    
        if self.control_mode == "velocity":
            motor_vels = [self.max_velocity for i in range(self.num_dof-len(self.free_dofs))]
            self.motor_out = to_torch(motor_vels, device=self.device)
        if self.control_mode == "effort":
            motor_efforts = [self.max_effort for i in range(self.num_dof-len(self.free_dofs))]
            self.motor_out = to_torch(motor_efforts, device=self.device)
        
        self.ant_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            robot_handle = self.gym.create_actor(env_ptr, ant_asset, start_pose, "twip", i, 1, 0)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, robot_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.ant_handles.append(robot_handle)

            dof_prop = self.gym.get_actor_dof_properties(env_ptr, robot_handle)
            if self.control_mode == "velocity":
                dof_prop["driveMode"][:] = gymapi.DOF_MODE_VEL
                dof_prop["stiffness"].fill(0.0)
                dof_prop["damping"].fill(600.0)
            if self.control_mode == "effort":
                dof_prop["driveMode"][:] = gymapi.DOF_MODE_EFFORT
                dof_prop["stiffness"].fill(0.0)
                dof_prop["damping"].fill(0.0)
            dof_prop["driveMode"][self.free_dofs] = gymapi.DOF_MODE_NONE
            dof_prop["stiffness"][self.free_dofs] = 0.0
            dof_prop["damping"][self.free_dofs] = 0.0
            for j in range(self.num_dof):
                if dof_prop['lower'][j] > dof_prop['upper'][j]:
                    self.dof_limits_lower.append(dof_prop['upper'][j])
                    self.dof_limits_upper.append(dof_prop['lower'][j])
                else:
                    self.dof_limits_lower.append(dof_prop['lower'][j])
                    self.dof_limits_upper.append(dof_prop['upper'][j])
            self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_prop)
        
    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_twip_reward(
            self.obs_buf, self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length
        )
        # print("Reset: ", torch.sum(self.reset_buf))

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        
        #self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        #NaN Check
        # print("NaN: ", torch.sum(torch.isnan(self.initial_root_states)))

        self.obs_buf[env_ids,0:4] = self.body_ori[env_ids,self.imu_frame_index].squeeze() #chassis orientation
        self.obs_buf[env_ids,4:7] = self.body_linvel[env_ids,self.imu_frame_index].squeeze() #chassis linvel

        return self.obs_buf

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        positions[:, self.free_dofs] = 0
        velocities = torch_rand_float(-0.25, 0.25, (len(env_ids), self.num_dof), device=self.device)
        velocities[:, self.free_dofs] = 0
        
        self.dof_pos[env_ids] = positions
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        dof_idx = list(range(self.num_dof))
        for i in self.free_dofs:
            dof_idx.remove(i)
        if self.control_mode == "velocity":
            vels = torch.zeros((self.actions.shape[0],self.num_dof)).to(self.device)
            #vels = torch.ones_like(self.actions) * self.motor_out
            vels[:,dof_idx] = self.actions * self.motor_out
            vel_tensor = gymtorch.unwrap_tensor(vels)
            self.gym.set_dof_velocity_target_tensor(self.sim, vel_tensor)
        if self.control_mode == "effort":
            forces = torch.zeros((self.actions.shape[0],self.num_dof)).to(self.device)
            #forces = torch.ones_like(self.actions) * self.motor_out * self.gear_ratio
            forces[:, dof_idx] = self.actions * self.motor_out#  * self.gear_ratio
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_actor_root_state_tensor(self.sim)

            points = []
            colors = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                # points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                #                glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                #                glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
                # colors.append([0.97, 0.1, 0.06])
                # points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(), glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                #                glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
                # colors.append([0.05, 0.99, 0.04])

            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_twip_reward(obs_buf, reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    ori_x = obs_buf[:, 0]
    ori_y = obs_buf[:, 1]
    ori_z = obs_buf[:, 2]
    ori_w = obs_buf[:, 3]
    vel_x = obs_buf[:, 4]
    vel_y = obs_buf[:, 5]
    vel_z = obs_buf[:, 6]

    ori = torch.stack((ori_x,ori_y,ori_z,ori_w)).transpose(0,1)
    vel = torch.stack((vel_x,vel_y,vel_z)).transpose(0,1)

    num_envs = ori.shape[0]
    #vertical = torch.tensor((-0.707107, 0.0, 0.0, 0.707107),device=ori_x.device).repeat(num_envs,1)
    #vertical = torch.tensor((0.0, 0.0, 0.0, 1.0),device=ori_x.device).repeat(num_envs,1)
    #pole_angle = torch.bmm(ori.view(num_envs, 1, 4), vertical.view(num_envs, 4, 1)).view(num_envs) 
    
    # We want to look at the pitch to determine reward/reset
    x = torch.atan2(2*(ori_w*ori_x+ori_y*ori_z),1-2*(ori_x**2+ori_y**2)) #pitch (0 when vertical)
    # y = torch.asin(2*(ori_w*ori_y-ori_x*ori_z)) #yaw
    # z = torch.atan2(2*(ori_w*ori_z+ori_y*ori_x),1-2*(ori_y**2+ori_z**2)) #roll
    pole_angle = 1 - torch.abs(x)

    cart_vel = torch.norm(vel, 2, 1)

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    #reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
    reward = pole_angle + 0.1 * torch.abs(cart_vel)

    # adjust reward for reset agents
    #reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    #reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(pole_angle) < 0.77, torch.ones_like(reward) * -2.0, reward)

    #reset = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf)
    #reset = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reset_buf), reset)
    reset = torch.where(torch.abs(pole_angle) < 0.77, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset
