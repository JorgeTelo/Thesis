import math
from itertools import combinations
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import tf_inverse, tf_combine
from copy import copy

import numpy as np
import torch

from base import BaseSim

class MySim(BaseSim):
    def __init__(self, args):
        super().__init__(args)

    def align_random_axis(self):
        ## Choose random axis of object to align
        axes = [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]]

        axis_idx = np.random.randint(len(axes))
        n = gymapi.Vec3(axes[axis_idx][0], axes[axis_idx][1], axes[axis_idx][2])

        joints_vecs = [gymapi.Vec3(self._mftip_state[0, 0],
                                   self._mftip_state[0, 1],
                                   self._mftip_state[0, 2]),
                       gymapi.Vec3(self._thtip_state[0, 0],
                                   self._thtip_state[0, 1],
                                   self._thtip_state[0, 2])]

        p = (joints_vecs[0] - joints_vecs[1]).normalize()

        ## Compute position of object as middle point between fingertips
        pose = gymapi.Transform()

        sum_vecs = gymapi.Vec3(0., 0., 0.)

        for i in range(2):
            sum_vecs += joints_vecs[i]

        pose.p = sum_vecs / 2

        ## Compute new rotation for object from axis-angle
        angle = np.arccos(n.dot(p) / n.length() * p.length()) 
        axis = n.cross(p).normalize()
        pose.r = gymapi.Quat().from_axis_angle(axis, angle)
    
        return pose

    def set_object_state(self):
        pose = self.align_random_axis()

        object_rigid_state = self.gym.get_actor_rigid_body_states(self.env, self.object_handle, gymapi.STATE_POS)
        
        object_rigid_state['pose']['p'][0][0] = pose.p.x
        object_rigid_state['pose']['p'][0][1] = pose.p.y
        object_rigid_state['pose']['p'][0][2] = pose.p.z

        object_rigid_state['pose']['r'][0][0] = pose.r.x
        object_rigid_state['pose']['r'][0][1] = pose.r.y
        object_rigid_state['pose']['r'][0][2] = pose.r.z
        object_rigid_state['pose']['r'][0][3] = pose.r.w

        self.gym.set_actor_rigid_body_states(self.env, self.object_handle, object_rigid_state, gymapi.STATE_POS)

def subscribe_events():
    my_sim.gym.subscribe_viewer_keyboard_event(my_sim.viewer, gymapi.KEY_X, "x")
    my_sim.gym.subscribe_viewer_keyboard_event(my_sim.viewer, gymapi.KEY_C, "c")
    my_sim.gym.subscribe_viewer_keyboard_event(my_sim.viewer, gymapi.KEY_V, "v")
    my_sim.gym.subscribe_viewer_keyboard_event(my_sim.viewer, gymapi.KEY_R, "r")

def torch_to_transform(q, p):
    pose = gymapi.Transform()
    pose.p.x = p[0][0]
    pose.p.y = p[0][1]
    pose.p.z = p[0][2]
    pose.r.x = q[0][0]
    pose.r.y = q[0][1]
    pose.r.z = q[0][2]
    pose.r.w = q[0][3]
    return pose

def transform_to_torch(t):
    return torch.tensor([[t.p.x, t.p.y, t.p.z, t.r.x, t.r.y, t.r.z, t.r.w]])

def transform_to_np(t):
    return np.array([[t.p.x, t.p.y, t.p.z, t.r.x, t.r.y, t.r.z, t.r.w]])

def transform_hand_pose(new_object_pose, original_object_pose):
    ohp = transform_to_torch(original_hand_pose)
    obp = transform_to_torch(original_object_pose)

    obp_inv_q, obp_inv_p = tf_inverse(new_object_pose[:, 3:7], new_object_pose[:, :3])

    t1_q, t1_p = tf_combine(obp_inv_q, obp_inv_p, ohp[:, 3:7], ohp[:, :3])

    t2_q, t2_p = tf_combine(obp[:, 3:7], obp[:, :3], t1_q, t1_p)

    return torch_to_transform(t2_q, t2_p)

args = gymutil.parse_arguments(
    description="Shadow Hand",
    custom_parameters=[
        {"name": "--exp", "type": str, "default": None, "help": ""}])

def load_data(data_folder):
    grasp_postures = np.load(data_folder + 'shadow_data.npy')
    grasp_labels = np.load(data_folder + 'shadow_labels.npy')
    return grasp_postures, grasp_labels

data_folder = './data/'
grasp_postures, grasp_labels = load_data(data_folder)

my_sim = MySim(args)

subscribe_events()

sphere_pose = gymapi.Transform()
sphere_geom = gymutil.WireframeSphereGeometry(0.01, 15, 15, sphere_pose, color=(1, 0, 0))

for i in range(grasp_postures.shape[0]):
    print (i, grasp_labels[i])

init_grasp_idx = 297
trgt_grasp_idx = 312

init_grasp = None
trgt_grasp = None

# Simulate
while not my_sim.gym.query_viewer_has_closed(my_sim.viewer):
    for evt in my_sim.gym.query_viewer_action_events(my_sim.viewer):
        if evt.action == "x" and evt.value > 0:

            pos = gymapi.Vec3(my_sim._fftip_state[0, 0],
                              my_sim._fftip_state[0, 1],
                              my_sim._fftip_state[0, 2])
            target_pos = gymapi.Transform(pos)

            gymutil.draw_lines(sphere_geom, my_sim.gym, my_sim.viewer, my_sim.env, target_pos)

            my_sim.set_object_state()

        elif evt.action == "c" and evt.value > 0:
            init_grasp = grasp_postures[init_grasp_idx].reshape(1, -1)
            init_grasp = my_sim.transfer_shadow_joints(init_grasp)
            my_sim.set_hand_target_position(init_grasp[0])

        elif evt.action == "v" and evt.value > 0:
            trgt_grasp = grasp_postures[trgt_grasp_idx].reshape(1, -1)
            trgt_grasp = my_sim.transfer_shadow_joints(trgt_grasp)

            num_steps = 15
            for i in range(num_steps + 1):
                grasp = i / num_steps * np.array(trgt_grasp[0]) + (num_steps - i) / num_steps * np.array(init_grasp[0])
                my_sim.set_hand_target_position(list(grasp))
                my_sim.step_simulation()

        elif evt.action == "r" and evt.value > 0:
            my_sim.reset_hand_state()
            my_sim.reset_object_state()

    my_sim.step_simulation()

print('Done')

my_sim.gym.destroy_viewer(my_sim.viewer)
my_sim.gym.destroy_sim(my_sim.sim)
