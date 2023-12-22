import math
from itertools import combinations
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from copy import copy

import numpy as np
import torch

class BaseSim():
    def __init__(self, args):
        self.args = args

        # initialize gym
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()

        self.setup_simulation()
        self.set_zero_gravity()

        self.load_hand()
        self.load_object()

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(1, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(1, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(1, -1, 13)

        self.handles = {
            "fftip": self.gym.find_actor_rigid_body_handle(self.env,
                                                           self.shadow_handle,
                                                           "fftip"),
            "mftip": self.gym.find_actor_rigid_body_handle(self.env,
                                                           self.shadow_handle,
                                                           "mftip"),
            "thtip": self.gym.find_actor_rigid_body_handle(self.env,
                                                           self.shadow_handle,
                                                           "thtip"),
        }

        self._fftip_state = self._rigid_body_state[:, self.handles["fftip"], :]
        self._thtip_state = self._rigid_body_state[:, self.handles["thtip"], :]
        self._mftip_state = self._rigid_body_state[:, self.handles["mftip"], :]
        self._object_state = self._root_state[:, self.object_handle, :]

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def setup_simulation(self):
        self.sim_params.substeps = 5
        self.sim_params.dt = 1.0 / 60.0

        self.sim_params.physx.num_position_iterations = 6
        self.sim_params.physx.num_velocity_iterations = 5
        self.sim_params.physx.bounce_threshold_velocity = 0.28
        self.sim_params.physx.contact_offset = 0.035
        self.sim_params.physx.rest_offset = 0.00001
        self.sim_params.physx.friction_offset_threshold = 0.01
        self.sim_params.physx.friction_correlation_distance = 0.05
        self.sim_params.physx.max_depenetration_velocity = 1000.0
        self.sim_params.physx.num_threads = self.args.num_threads
        self.sim_params.physx.use_gpu = self.args.use_gpu
        self.sim_params.use_gpu_pipeline = False
        self.sim_params.up_axis = gymapi.UP_AXIS_Z

        self.sim = self.gym.create_sim(self.args.compute_device_id,
                self.args.graphics_device_id, self.args.physics_engine,
                self.sim_params)

        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise ValueError('*** Failed to create viewer')

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # set up the env grid
        num_envs = 1
        spacing = 1.5
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, 0.0, spacing)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_envs)

        # Look at the first env
        cam_pos = gymapi.Vec3(0, 2, 1.5)
        cam_target = gymapi.Vec3(0, 0.2, .0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def step_simulation(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)
        self._refresh()

    def load_hand(self):
        pose = gymapi.Transform()
        pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        asset_root = "assets"

        asset_file = "urdf/shadow_hand_description/shadowhand_with_fingertips_collision_2.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.thickness = 0.0001
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True

        print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
        shadow_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.shadow_handle = self.gym.create_actor(self.env, shadow_asset, pose, "shadow", 0, 1)
        shadow_dof_props = self.gym.get_actor_dof_properties(self.env, self.shadow_handle)
        self.shadow_num_dofs = len(shadow_dof_props)

        shadow_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        # override default stiffness and damping values
        shadow_dof_props['stiffness'].fill(400.0)
        shadow_dof_props['damping'].fill(200.0)
        shadow_dof_props['effort'][0:2] = 500.0
        shadow_dof_props['effort'][3:] = 5.0
        shadow_dof_props['velocity'][3:] = 0.5
        # set new actor properties
        self.gym.set_actor_dof_properties(self.env, self.shadow_handle, shadow_dof_props)

        self.reset_hand_state()

    def load_object(self):
        asset_root = "assets"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = False
        asset_options.thickness = 0.0001
        asset_options.disable_gravity = False

        object_asset = self.gym.create_box(self.sim, 0.060, 0.030, 0.030, asset_options)
        # Object asset pose
        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(0.0, 0.2, 0.6)
        object_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        self.object_handle = self.gym.create_actor(self.env, 
                                                   object_asset,
                                                   object_pose, 
                                                   "object", 0, 0)

    def transfer_shadow_joints (self, joint_angles):
        new_joint_angles = []
        for i in range(joint_angles.shape[0]):
            new_joint_angle = [0] * 24
            joint_map = [5, 4, 3, 2, 14, 13, 12, 11, 18, 17, 16, 15, 10, 9, 8, 7, 23, 22, 21, 20, 19]
            for j, v in enumerate(joint_angles[i]):
                new_joint_angle[joint_map[j]] = (v * np.pi) / 180.0
            new_joint_angles.append(new_joint_angle)

        return new_joint_angles

    def set_hand_pose(self, pose):
        rb_state = self.gym.get_actor_rigid_body_states(self.env, self.shadow_handle, gymapi.STATE_POS)
        rb_state['pose']['p'][0][0] = pose[0]
        rb_state['pose']['p'][0][1] = pose[1]
        rb_state['pose']['p'][0][2] = pose[2]
        rb_state['pose']['r'][0][0] = pose[3]
        rb_state['pose']['r'][0][1] = pose[4]
        rb_state['pose']['r'][0][2] = pose[5]
        rb_state['pose']['r'][0][3] = pose[6]
        self.gym.set_actor_rigid_body_states(self.env, self.shadow_handle, rb_state, gymapi.STATE_POS)

    def reset_hand_pose(self):
        shadow_pose = np.array([0.0, 0.0, 0.0, -0.707107, 0.0, 0.0, 0.707107])
        self.set_hand_pose(shadow_pose)

    def set_hand_state(self, state):
        shadow_dof_states = self.gym.get_actor_dof_states(self.env, self.shadow_handle, gymapi.STATE_ALL)

        for j in range(self.shadow_num_dofs):
            shadow_dof_states['pos'][j] = state[j]

        self.gym.set_actor_dof_states(self.env, self.shadow_handle, shadow_dof_states, gymapi.STATE_POS)
        self.gym.set_actor_dof_position_targets(self.env, self.shadow_handle, state)

    def reset_hand_state(self, is_lateral=False):
        shadow_state = [0] * self.shadow_num_dofs

        if is_lateral:
            shadow_state[19] = -0.6
            shadow_state[20] = 0.68
        else:
            shadow_state[20] = 1.58

        self.set_hand_state(shadow_state)

    def set_hand_target_position(self, target_pos):
        self.gym.set_actor_dof_position_targets(self.env, self.shadow_handle, target_pos)

    def set_object_rotation(self, state):
        rb_state = self.gym.get_actor_rigid_body_states(self.env, self.object_handle, gymapi.STATE_ALL)
        rb_state['pose']['r'][0][0] = state[0]
        rb_state['pose']['r'][0][1] = state[1]
        rb_state['pose']['r'][0][2] = state[2]
        rb_state['pose']['r'][0][3] = state[3]
        self.gym.set_actor_rigid_body_states(self.env, self.object_handle, rb_state, gymapi.STATE_POS)

    def set_object_position(self, state):
        rb_state = self.gym.get_actor_rigid_body_states(self.env, self.object_handle, gymapi.STATE_ALL)
        rb_state['pose']['p'][0][0] = state[0]
        rb_state['pose']['p'][0][1] = state[1]
        rb_state['pose']['p'][0][2] = state[2]
        self.gym.set_actor_rigid_body_states(self.env, self.object_handle, rb_state, gymapi.STATE_POS)

    def set_object_pose(self, state):
        self.set_object_position(state[:3])
        self.set_object_rotation(state[3:7])

    def reset_object_state(self):
        default_object_pose = np.array([0.0, 1.0, 0.0, -0.707107, 0.0, 0.0, 0.707107])
        self.set_object_pose(default_object_pose)
        self.reset_object_velocity()

    def reset_object_velocity(self):
        rb_state = self.gym.get_actor_rigid_body_states(self.env, self.object_handle, gymapi.STATE_ALL)
        rb_state['vel']['linear'][0][0] = 0
        rb_state['vel']['linear'][0][1] = 0
        rb_state['vel']['linear'][0][2] = 0
        rb_state['vel']['angular'][0][0] = 0
        rb_state['vel']['angular'][0][1] = 0
        rb_state['vel']['angular'][0][2] = 0
        self.gym.set_actor_rigid_body_states(self.env, self.object_handle, rb_state, gymapi.STATE_POS)

    def draw_poses(self, poses, axis_size=0.1):
        self.gym.clear_lines(self.viewer)
        axis_geom = gymutil.AxesGeometry(axis_size)

        for i in range(poses.shape[0]):
            axis_pose = gymapi.Transform()
            axis_pose.p.x = poses[i][0]
            axis_pose.p.y = poses[i][1]
            axis_pose.p.z = poses[i][2]

            if poses.shape[1] == 7:
                axis_pose.r.x = poses[i][3]
                axis_pose.r.y = poses[i][4]
                axis_pose.r.z = poses[i][5]
                axis_pose.r.w = poses[i][6]

            gymutil.draw_lines(axis_geom, self.gym, self.viewer, self.env, axis_pose)

    def set_zero_gravity(self):
        gravity_vec = gymapi.Vec3(0.0, 0.0, 0.0)
        self.__set_gravity(gravity_vec)
        print ("Gravity set to zero!")

    def set_earth_gravity(self):
        gravity_vec = gymapi.Vec3(0.0, -9.8, 0.0)
        self.__set_gravity(gravity_vec)
        print ("Gravity set to -9.8!")

    def __set_gravity(self, gravity_vec):
        self.sim_params.gravity = gravity_vec
        self.gym.set_sim_params(self.sim, self.sim_params)

