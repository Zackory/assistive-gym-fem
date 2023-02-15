#%%
import os
import pickle

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
import math

#%%
DEFAULT_body_info = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'body_info.pkl'),'rb'))


limb_config = {
    'hand':[2],
    'forearm':[6, 5],
    'upperarm':[7, 5],
    'foot':[2],
    'shin':[7, 5],
    'thigh':[7, 5],
    'upperchest':[4],
    'head':[3]
}

# body segments to be covered with points as defined by the joints in the observation
# key = value for the joint in assistive_gym
# value = indices of human pose (from obs) that give the x,y pose of the joints that define the body segment
obs_limbs = {
    18:[[0], 'hand'],           # right hand: wrist
    16:[[0,1], 'forearm'],      # right forearm: wrist -> elbow
    14:[[1,2], 'upperarm'],     # right upperarm: elbow -> shoulder
    39:[[3], 'foot'],           # right foot: ankle
    36:[[3,4], 'shin'],         # right shin: ankle -> knee
    35:[[4,5], 'thigh'],        # right thigh: knee -> hip
    28:[[6], 'hand'],           # left hand: wrist
    26:[[6,7], 'forearm'],      # left forearm: wrist -> elbow
    24:[[7,8], 'upperarm'],     # left upperarm: elbow -> shoulder
    46:[[9], 'foot'],           # left foot: ankle
    43:[[9,10], 'shin'],        # left shin: ankle -> knee
    42:[[10,11], 'thigh'],      # left thigh: knee -> hip
    8:[[12], ('upperchest', 'waist')],     # upperchest
    32:[[13], 'head']}          # head

# indices of human_pose (from obs) that give the x,y pose of the joints that define the target body part
# index to select target limb
# 0: right hand, 1: right forearm, 2: right upperarm
# 3: right foot, 4: right shin, 5: right thigh
# 6: left hand, 7: left forearm, 8: left upperarm
# 9: left foot, 10: left shin, 11: left thigh
# 12: both shins, 13: whole torso, 14: both legs
# 15: whole body
obs_limbs_list = list(obs_limbs.keys())
all_possible_target_limbs = [
                [obs_limbs_list[0]], obs_limbs_list[0:2], obs_limbs_list[0:3], 
                [obs_limbs_list[3]], obs_limbs_list[3:5], obs_limbs_list[3:6],
                [obs_limbs_list[6]], obs_limbs_list[6:8], obs_limbs_list[6:9],
                [obs_limbs_list[9]], obs_limbs_list[9:11], obs_limbs_list[9:12],
                [39, 36, 46, 43], [18, 16, 14, 28, 26, 24, 2, 8], [39, 36, 46, 43, 35, 42],
                [18, 16, 14, 39, 36, 35, 28, 26, 24, 46, 43, 42, 8]]
target_limb_subset = [2, 4, 5, 8, 10, 11, 12, 13, 14, 15]
# all_possible_target_limbs = [all_possible_target_limbs[i] for i in [2, 4, 5, 8, 10, 11, 12, 13, 14, 15]]
# all_possible_target_limbs



#%%
def get_rectangular_limb_points(point1, point2, capsule_radius=None, length_points=0, width_points=0):
    axis_vector = point1-point2

    quadrant_adjustment = 0
    if axis_vector[1] < 0 and axis_vector[0] >= 0:
        quadrant_adjustment = np.pi
    elif axis_vector[1] < 0 and axis_vector[0] < 0:
        quadrant_adjustment = -np.pi

    theta = np.arctan(axis_vector[0]/axis_vector[1]) + quadrant_adjustment

    length = np.linalg.norm(point1 - point2)
    x = np.linspace(-capsule_radius, capsule_radius, width_points)
    y = np.linspace(0, length, length_points+1)
    X, Y = np.meshgrid(x, y)
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    grid = np.concatenate((X, Y), axis=1)

    r = np.array(( (np.cos(theta), np.sin(theta)),
               (-np.sin(theta),  np.cos(theta)) ))
    grid_r = []
    for row in grid:
        grid_r.append(r.dot(row)+point2)

    grid_r = np.array(grid_r)

    return grid_r[:-width_points,:]

# https://stackoverflow.com/questions/33510979/generator-of-evenly-spaced-points-in-a-circle-in-python
def get_circular_limb_points(point, radius=None, num_rings=None):
    r = np.linspace(0, radius, num_rings)
    n = [1+(x*5) for x in range(num_rings)]
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circle = np.c_[x, y] + point
        circles.append(circle)
    circles = np.concatenate(circles)
    return circles


def get_torso_points(human_pose, radius_upperchest, radius_waist, num_rings):
    shoulder_midpoint = (human_pose[2] + human_pose[8])/2
    hip_midpoint = (human_pose[5] + human_pose[11])/2
    # print(radius_upperchest, radius_waist)
    
    calc_chest = (human_pose[12] + shoulder_midpoint)/2
    calc_waist = (human_pose[12] + hip_midpoint)/2

    offset = 0.005 # subtract this offest to prevent overlap of torso points with hip and shoulder points
    chest_points = get_circular_limb_points(calc_chest, radius=radius_upperchest-offset, num_rings=num_rings)
    waist_points = get_circular_limb_points(calc_waist, radius=radius_waist-offset, num_rings=num_rings)

    return np.concatenate((chest_points, waist_points))


def get_body_points_from_obs(human_pose, target_limb_code, body_info = None):
    # global obs_limbs, limb_config, all_possible_target_limbs

    # TESTING ONLY: Move the human around so blanket covers different limbs
    # for i in range(len(human_pose)):
    #     human_pose[i, 1] += 0.15
    if body_info is None:
        body_info = DEFAULT_body_info

    target_points = []
    nontarget_points = []
    all_body_points = []
    for limb, limb_info in obs_limbs.items():
        joints = limb_info[0]
        limb_name = limb_info[1]
        if limb == 8:
            limb_points = get_torso_points(human_pose, radius_upperchest=body_info[limb_name[0]][1], radius_waist=body_info[limb_name[1]][1], num_rings=limb_config[limb_name[0]][0])
        elif len(joints) == 1:
            limb_points = get_circular_limb_points(
                human_pose[joints[0]], radius=body_info[limb_name][1], num_rings=limb_config[limb_name][0])
        else:
            limb_points = get_rectangular_limb_points(
                human_pose[joints[0]], human_pose[joints[1]],
                capsule_radius=body_info[limb_name][1], length_points=limb_config[limb_name][0], width_points=limb_config[limb_name][1])

        num_limb_points = len(limb_points)
        is_target = np.ones((num_limb_points, 1)) if limb in all_possible_target_limbs[target_limb_code] else np.zeros((num_limb_points, 1))
        is_target = is_target - 1 if limb is 32 else is_target # if the point is on the head, val is -1
        all_body_points.append(np.hstack((limb_points, is_target)))
    
    return np.concatenate(all_body_points)
        # if limb in all_possible_target_limbs[target_limb_code]:
        #     target_points.append(limb_points)
        # else:
        #     nontarget_points.append(limb_points)
    # return 0, np.concatenate(nontarget_points)
    # return np.concatenate(target_points), np.concatenate(nontarget_points)

#%%
def sub_sample_point_clouds(cloth_initial_3D_pos, cloth_final_3D_pos, voxel_size = 0.05):
    "https://towardsdatascience.com/how-to-automate-lidar-point-cloud-processing-with-python-a027454a536c"

    cloth_initial = np.array(cloth_initial_3D_pos)
    cloth_final = np.array(cloth_final_3D_pos)
    nb_vox=np.ceil((np.max(cloth_initial, axis=0) - np.min(cloth_initial, axis=0))/voxel_size)
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((cloth_initial - np.min(cloth_initial, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    idx_pts_vox_sorted=np.argsort(inverse)

    voxel_grid={}
    voxel_grid_cloth_inds={}
    cloth_initial_subsample=[]
    cloth_final_subsample = []
    last_seen=0
    for idx,vox in enumerate(non_empty_voxel_keys):
        voxel_grid[tuple(vox)]= cloth_initial[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
        voxel_grid_cloth_inds[tuple(vox)] = idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]
        
        closest_point_to_barycenter = np.linalg.norm(voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()
        cloth_initial_subsample.append(voxel_grid[tuple(vox)][closest_point_to_barycenter])
        cloth_final_subsample.append(cloth_final[voxel_grid_cloth_inds[tuple(vox)][closest_point_to_barycenter]])

        last_seen+=nb_pts_per_voxel[idx]
    # print(len(cloth_initial_subsample), len(cloth_final_subsample))
    
    return cloth_initial_subsample, cloth_final_subsample

#%%
# improved performance by using numpy operations instead of looping over the points in the array
def get_covered_status(all_body_points, cloth_state_2D):

        covered_threshold = 0.05
        covered_status = []

        for body_point in all_body_points:
            is_covered = False
            if np.any(np.linalg.norm(cloth_state_2D - body_point[0:2], axis=1) <= covered_threshold):
                is_covered = True
            is_target = body_point[2] # 1 = target, 0 = nontarget, -1 = head
            covered_status.append([is_target, is_covered]) 
        
        covered_status = get_closest_t_to_nt_points(all_body_points, covered_status)
        return covered_status

#%%
def get_closest_t_to_nt_points(all_body_points, covered_status):
    # for i in range(len(all_body_points)):
    #     if
    # nt_points_2D = all_body_points[covered_status[:,0] == 0][:, 0:2]
    covered_status = np.array(covered_status)
    t_points_2D = all_body_points[covered_status[:,0] == 1][:, 0:2]

    min_dists = []
    # for nt_point in nt_points_2D:
    for i in range(len(all_body_points)):
        if covered_status[i,0] == 0:
            nt_point = all_body_points[i]
            min_dist_to_t_point = np.min(np.linalg.norm(t_points_2D - nt_point[0:2], axis=1))
            min_dists.append(min_dist_to_t_point)
        else:
            min_dists.append(np.nan)
    
    # norm_dist = np.nanmin(min_dists) # not consistent based on pose
    norm_dist = 0.05 #! need to find a way to keep this consistent and also adjust for different body sizes
    # print(norm_dist)

    num_t_points = len(t_points_2D)
    num_nt_points = len(all_body_points)-num_t_points
    weight_factor = num_nt_points/num_t_points
    # normalized_dist = np.array(min_dists)
    normalized_dist = np.array(min_dists)/(norm_dist*weight_factor)
    # print(np.array(min_dists).shape)
    # normalized_dist = abs((np.array(min_dists)-norm_dist)/norm_dist)
    # normalized_dist = ((np.array(min_dists)-norm_dist)/norm_dist)**2
    
    return np.insert(covered_status, 2, normalized_dist, axis=1)


#%%
def get_body_points_reward(all_body_points, cloth_initial_2D, cloth_final_2D):
    initially_covered_status = get_covered_status(all_body_points, cloth_initial_2D)
    covered_status = get_covered_status(all_body_points, cloth_final_2D)


    # head_ind = len(covered_status)-1
    target_uncovered_reward = 0
    nontarget_uncovered_penalty = 0
    head_covered_penalty = 0
    for i in range(len(covered_status)):
        is_target = covered_status[i][0]
        is_covered = covered_status[i][1]
        is_initially_covered = initially_covered_status[i][1]
        if is_target == 1 and not is_covered: # target uncovered reward
            target_uncovered_reward += 1
        elif is_target == -1 and is_covered and not is_initially_covered: # head covered penalty
            head_covered_penalty += 1
        elif is_target == 0 and not is_covered and is_initially_covered:
            nontarget_uncovered_penalty += covered_status[i][2]
            # nontarget_uncovered_penalty += 1
    # print(target_uncovered_reward, nontarget_uncovered_penalty, head_covered_penalty)
    num_target = np.count_nonzero(all_body_points[:,2] == 1)
    num_head = np.count_nonzero(all_body_points[:,2] == -1)
    target_uncovered_reward = 100*(target_uncovered_reward/num_target)
    nontarget_uncovered_penalty = -100*(nontarget_uncovered_penalty/num_target)
    head_covered_penalty = -200*(head_covered_penalty/num_head)
    reward = target_uncovered_reward + nontarget_uncovered_penalty + head_covered_penalty
    # print(target_uncovered_reward, nontarget_uncovered_penalty, head_covered_penalty)
    # if self.rendering:
    #     print('initial covered', initially_covered_status)
    #     print('covered', covered_status)
    #     print(target_uncovered_reward, nontarget_uncovered_penalty, head_covered_penalty)
    #     print(reward)
    # self.covered_status = covered_status

    return reward, covered_status

def get_reward(action, all_body_points, cloth_initial_2D, cloth_final_2D):
    reward_distance_btw_grasp_release = -150 if np.linalg.norm(action[0:2] - action[2:]) >= 1.5 else 0
    body_point_reward, covered_status = get_body_points_reward(all_body_points, cloth_initial_2D, cloth_final_2D)
    reward = body_point_reward + reward_distance_btw_grasp_release
    return reward, covered_status

#%%
def randomize_target_limbs(tl_subset=target_limb_subset):
    # target_limb_code = np.random.randint(len(all_possible_target_limbs))
    target_limb_code = np.random.choice(tl_subset)
    return target_limb_code

# new possible method - maybe this will work better for PPO?
def remap_action_ppo(action, remap_ranges):
    remap_action = []
    for i in range(len(action)):
        a = np.interp(action[i], [-1, 1], remap_ranges[i])
        remap_action.append(a)
    return np.array(remap_action)

def scale_action(action, scale=[0.44, 1.05]):
    scale = scale*2
    return scale*action

# clipping threshold 0.028 if not subsampled gt cloth points, 0.05
def check_grasp_on_cloth(action, cloth_initial, clipping_thres=0.028):
    grasp_loc = action[0:2]
    dist = np.linalg.norm(cloth_initial[:,0:2] - grasp_loc, axis=1)
    # * if no points on the blanket are within 2.8 cm of the grasp location, clip
    is_on_cloth = (np.any(np.array(dist) < clipping_thres)) 
    return dist, is_on_cloth

#%%
def get_edge_connectivity(cloth_initial, edge_threshold, cloth_dim):
    """
    returns an array of edge indexes, returned as a list of index tuples
    Data requires indexes to be in COO format so will need to convert via performing transpose (.t()) and calling contiguous (.contiguous())
    """
    cloth_initial = np.array(cloth_initial)
    if cloth_dim == 2:
        cloth_initial = np.delete(cloth_initial, 2, axis = 1)
    threshold = edge_threshold
    edge_inds = []
    for p1_ind, point_1 in enumerate(cloth_initial):
        for p2_ind, point_2 in enumerate(cloth_initial): # want duplicate edges to capture both directions of info sharing
            if p1_ind != p2_ind and np.linalg.norm(point_1 - point_2) <= threshold: # don't consider distance between a point and itself, see if distance is within
                edge_inds.append([p1_ind, p2_ind])
            np.linalg.norm(point_1 - point_2) <= threshold
    # return torch.tensor([0,2], dtype = torch.long)
    return torch.tensor(edge_inds, dtype = torch.long)

def get_rotation_matrix(axis, theta):
    """
    Find the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians.
    Credit: http://stackoverflow.com/users/190597/unutbu

    Args:
        axis (list): rotation axis of the form [x, y, z]
        theta (float): rotational angle in radians

    Returns:
        array. Rotation matrix.
    """

    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                    [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                    [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


# #%%
# # ## TEST VISUALIZATION OF BODY POINTS
# import matplotlib.lines as mlines
# import time
# t0 = time.time()

# # filename_env = '/home/kpputhuveetil/git/vBM-GNNdev/cmaes_eval_500_new_rew/tl0_c23_13938182822138537765_pid14359.pkl'
# # filename_env = '/home/kpputhuveetil/git/vBM-GNNs/c0_10519595811781955081_pid5411.pkl'
# # filename_env = '/home/kpputhuveetil/git/vBM-GNNs/c0_10109917428862267910_pid41377.pkl'
# filename_env = '/home/kpputhuveetil/git/vBM-GNNdev/gnn_new_data/raw/c0_24035249859464233_pid95651.pkl'
# target_limb_code = 4
# raw_data = pickle.load(open(filename_env,'rb'))
# human_pose = np.reshape(raw_data['observation'][0], (-1,2))
# action = raw_data['action']

# # human_pose = raw_data['human_pose']


# all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code)


# cloth_initial_subsample, cloth_final_subsample = sub_sample_point_clouds(raw_data['info']['cloth_initial'][1], raw_data['info']['cloth_final'][1])
# # cloth_initial_subsample, cloth_final_subsample = sub_sample_point_clouds(raw_data['sim_info']['info']['cloth_initial'][1], raw_data['sim_info']['info']['cloth_final'][1])
# cloth_initial = np.delete(np.array(cloth_initial_subsample), 2, axis = 1)
# cloth_final = np.delete(np.array(cloth_final_subsample), 2, axis = 1)
# # cloth_final = raw_data['cma_info']['best_pred']

# reward, covered_status = get_body_points_reward(all_body_points, cloth_initial, cloth_final)

# print('Reward:',reward)

# point_colors = []
# for point in covered_status:
#     is_target = point[0]
#     is_covered = point[1]
#     if is_target == 1:
#         color = 'purple' if is_covered else 'forestgreen'
#     elif is_target == -1:
#         color = 'red' if is_covered else 'darkorange'
#     else:
#         color = 'darkorange' if is_covered else 'red'
#     point_colors.append(color)

# plt.figure(figsize=[4, 6])
# plt.scatter(all_body_points[:,0], all_body_points[:,1], c=point_colors)
# # plt.scatter(human_pose[:,0], human_pose[:,1], c='navy')

# ntarg = mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None', label='uncovered points')
# targ = mlines.Line2D([], [], color='forestgreen', marker='o', linestyle='None', label='target points')
# obs = mlines.Line2D([], [], color='navy', marker='o', linestyle='None', label='observation points')
# plt.scatter(cloth_final[:,0], cloth_final[:,1], alpha=0.2)
# plt.axis([-0.7, 0.7, -1.0, 0.9])
# # plt.legend(loc='lower left', prop={'size': 9}, handles=[ntarg, targ, obs])
# # plt.gca().invert_yaxis()
# plt.show()
# print('Time:', time.time()-t0)



# # %%
