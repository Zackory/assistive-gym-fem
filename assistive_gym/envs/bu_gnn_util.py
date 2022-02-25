#%%
import pickle, os
import numpy as np
import matplotlib.pyplot as plt

#%%
body_info = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'body_info.pkl'),'rb'))


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
    8:[[12], 'upperchest'],     # upperchest
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



#%%
def get_rectangular_limb_points(point1, point2, capsule_radius=None, length_points=0, width_points=0):
    axis_vector = point1-point2
    theta = np.arctan(axis_vector[0]/axis_vector[1])

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

#%%
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


#%%
def get_body_points_from_obs(human_pose, target_limb_code):
    # global obs_limbs, limb_config, all_possible_target_limbs

    target_points = []
    nontarget_points = []
    all_points = []
    for limb, limb_info in obs_limbs.items():
        joints = limb_info[0]
        limb_name = limb_info[1]
        if len(joints) == 1:
            limb_points = get_circular_limb_points(
                human_pose[joints[0]], radius=body_info[limb_name][1], num_rings=limb_config[limb_name][0])
        else:
            limb_points = get_rectangular_limb_points(
                human_pose[joints[0]], human_pose[joints[1]],
                capsule_radius=body_info[limb_name][1], length_points=limb_config[limb_name][0], width_points=limb_config[limb_name][1])

        num_limb_points = len(limb_points)
        is_target = np.ones((num_limb_points, 1)) if limb in all_possible_target_limbs[target_limb_code] else np.zeros((num_limb_points, 1))
        is_target = is_target - 1 if limb is 32 else is_target # if the point is on the head, val is -1
        all_points.append(np.hstack((limb_points, is_target)))
    
    return np.concatenate(all_points)
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
def get_covered_status(all_points, cloth_state_2D):
        covered_status = []

        for body_point in all_points.tolist():
            is_covered = False
            for cloth_point in cloth_state_2D:
                if np.linalg.norm(cloth_point - body_point[0:2]) <= 0.05:
                    is_covered = True
                    break
            is_target = body_point[2] # 1 = target, 0 = nontarget, -1 = head
            covered_status.append([is_target, is_covered]) 
        
        return covered_status

#%%
def get_body_points_reward(all_points, cloth_initial_2D, cloth_final_2D):
        initially_covered_status = get_covered_status(all_points, cloth_initial_2D)
        covered_status = get_covered_status(all_points, cloth_final_2D)


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
            elif is_target == -1 and is_covered: # head covered penalty
                head_covered_penalty = 1
            elif is_target == 0 and not is_covered and is_initially_covered:
                nontarget_uncovered_penalty += 1

        num_target = np.count_nonzero(all_points[:,2] == 1)
        num_head = np.count_nonzero(all_points[:,2] == -1)
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

#%%
# ## TEST VISUALIZATION OF BODY POINTS
# import matplotlib.lines as mlines

# filename_env = '/home/kpputhuveetil/git/vBM-GNNs/c0_10519595811781955081_pid5411.pkl'
# # filename_env = '/home/kpputhuveetil/git/vBM-GNNs/c0_10109917428862267910_pid41377.pkl'
# raw_data = pickle.load(open(filename_env,'rb'))
# human_pose = np.reshape(raw_data['observation'][0], (-1,2))
# all_points = get_body_points_from_obs(human_pose, 14)

# cloth_initial_subsample, cloth_final_subsample = sub_sample_point_clouds(raw_data['info']['cloth_initial'][1], raw_data['info']['cloth_final'][1])
# cloth_initial = np.delete(np.array(cloth_initial_subsample), 2, axis = 1)
# cloth_final = np.delete(np.array(cloth_final_subsample), 2, axis = 1)

# reward, covered_status = get_body_points_reward(all_points, cloth_initial, cloth_final)

# print(reward)

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
# plt.scatter(all_points[:,0], all_points[:,1], c=point_colors)
# plt.scatter(human_pose[:,0], human_pose[:,1], c='navy')

# ntarg = mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None', label='uncovered points')
# targ = mlines.Line2D([], [], color='forestgreen', marker='o', linestyle='None', label='target points')
# obs = mlines.Line2D([], [], color='navy', marker='o', linestyle='None', label='observation points')
# plt.scatter(cloth_final[:,0], cloth_final[:,1], alpha=0.2)
# plt.axis([-0.7, 0.7, -1.0, 0.9])
# # plt.legend(loc='lower left', prop={'size': 9}, handles=[ntarg, targ, obs])
# plt.gca().invert_yaxis()
# plt.show()


# %%
