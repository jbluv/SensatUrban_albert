from os.path import join, exists, dirname, abspath
import numpy as np
import colorsys, random, os, sys
import open3d as o3d
from helper_ply import read_ply, write_ply
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors


class ConfigSensatUrban:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    num_classes = 13  # Number of valid classes
    sub_grid_size = 0.2  # preprocess_parameter

    batch_size = 4  # batch_size during training
    val_batch_size = 4  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 50  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension for randlanet 
    # d_out = [8, 16, 32, 64, 128]  # feature dimension for point transformer

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log_SensatUrban'
    saving = True
    saving_path = None


class DataProcessing:

    @staticmethod
    def get_num_class_from_label(labels, total_class):
        num_pts_per_class = np.zeros(total_class, dtype=np.int32)
        # original class distribution
        val_list, counts = np.unique(labels, return_counts=True)
        for idx, val in enumerate(val_list):
            num_pts_per_class[val] += counts[idx]
        # for idx, nums in enumerate(num_pts_per_class):
        #     print(idx, ':', nums)
        return num_pts_per_class

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def read_ply_data(path, with_rgb=True, with_label=True):
        data = read_ply(path)
        xyz = np.vstack((data['x'], data['y'], data['z'])).T
        if with_rgb and with_label:
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            labels = data['class']
            return xyz.astype(np.float32), rgb.astype(np.uint8), labels.astype(np.uint8)
        elif with_rgb and not with_label:
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            return xyz.astype(np.float32), rgb.astype(np.uint8)
        elif not with_rgb and with_label:
            labels = data['class']
            return xyz.astype(np.float32), labels.astype(np.uint8)
        elif not with_rgb and not with_label:
            return xyz.astype(np.float32)

    @staticmethod
    def random_sub_sampling(points, features=None, labels=None, sub_ratio=10, verbose=0):
        num_input = np.shape(points)[0]
        num_output = num_input // sub_ratio
        idx = np.random.choice(num_input, num_output)
        if (features is None) and (labels is None):
            return points[idx]
        elif labels is None:
            return points[idx], features[idx]
        elif features is None:
            return points[idx], labels[idx]
        else:
            return points[idx], features[idx], labels[idx]

    @staticmethod
    def get_class_weights(num_per_class, name='sqrt'):
        # # pre-calculate the number of points in each category
        frequency = num_per_class / float(sum(num_per_class))
        if name == 'sqrt' or name == 'lovas':
            ce_label_weight = 1 / np.sqrt(frequency)
        elif name == 'wce':
            ce_label_weight = 1 / (frequency + 0.02)
        else:
            raise ValueError('Only support sqrt and wce')
        return np.expand_dims(ce_label_weight, axis=0)


class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            o3d.visualization.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])

        o3d.geometry.PointCloud.estimate_normals(pc)
        o3d.visualization.draw_geometries([pc], width=1000, height=1000)
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None):
        # only visualize a number of points to save memory
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            # ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=1)
            ins_colors = [[85, 107, 47],  # ground -> OliveDrab
                          [0, 255, 0],  # tree -> Green
                          [255, 165, 0],  # building -> orange
                          [41, 49, 101],  # Walls ->  darkblue
                          [0, 0, 0],  # Bridge -> black
                          [0, 0, 255],  # parking -> blue
                          [255, 0, 255],  # rail -> Magenta
                          [200, 200, 200],  # traffic Roads ->  grey
                          [89, 47, 95],  # Street Furniture  ->  DimGray
                          [255, 0, 0],  # cars -> red
                          [255, 255, 0],  # Footpath  ->  deeppink
                          [0, 255, 255],  # bikes -> cyan
                          [0, 191, 255]  # water ->  skyblue
                          ]

        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0]);
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1]);
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2]);
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        # Plot.draw_pc(Y_semins)
        print(Y_semins)
        return Y_semins

    @staticmethod
    def save_ply_o3d(data, save_name):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])
        if np.shape(data)[1] == 3:
            o3d.io.write_point_cloud(save_name, pcd)
        elif np.shape(data)[1] == 6:
            if np.max(data[:, 3:6]) > 20:
                pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.)
            else:
                pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6])
            o3d.io.write_point_cloud(save_name, pcd)
        return


def tf_augment_input(stacked_points, batch_inds):
    """
    Augment inputs with rotation, scale and noise
    """
    rot_type = "veritical"
    augment_scale_min = 0.7
    augment_scale_max = 1.3
    augment_symmetries = [True, False, False]
    augment_noise= 0.001
    
    # Parameter
    num_batches = batch_inds[-1] + 1

    # Rotation
    if rot_type == "veritical":
        print("veritical rotation")
        stacked_points = tf.reshape(stacked_points,(-1,3))
        theta = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)
        # Rotation matrices
        c, s = tf.cos(theta), tf.sin(theta)
        cs0 = tf.zeros_like(c)
        cs1 = tf.ones_like(c)
        # c -s  0
        # s  c  0
        # 0  0  1
        R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
        R = tf.reshape(R, (-1, 3, 3))
        # Create N x 3 x 3 rotation matrices to multiply with stacked_points
        stacked_rots = tf.gather(R, batch_inds)
        # Apply rotations
        stacked_points = tf.reshape(tf.matmul(tf.expand_dims(stacked_points, axis=1), stacked_rots), [-1, 3])

    elif rot_type == "arbitrary":
        print("arbitrary rotation")
        cs0 = tf.zeros((num_batches,))
        cs1 = tf.ones((num_batches,))
        # x rotation
        thetax = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)
        cx, sx = tf.cos(thetax), tf.sin(thetax)
        Rx = tf.stack([cs1, cs0, cs0, cs0, cx, -sx, cs0, sx, cx], axis=1)
        Rx = tf.reshape(Rx, (-1, 3, 3))
        # y rotation
        thetay = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)
        cy, sy = tf.cos(thetay), tf.sin(thetay)
        Ry = tf.stack([cy, cs0, -sy, cs0, cs1, cs0, sy, cs0, cy], axis=1)
        Ry = tf.reshape(Ry, (-1, 3, 3))
        # z rotation
        thetaz = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)
        cz, sz = tf.cos(thetaz), tf.sin(thetaz)
        Rz = tf.stack([cz, -sz, cs0, sz, cz, cs0, cs0, cs0, cs1], axis=1)
        Rz = tf.reshape(Rz, (-1, 3, 3))
        # whole rotation
        Rxy = tf.matmul(Rx, Ry)
        R = tf.matmul(Rxy, Rz)
        # Create N x 3 x 3 rotation matrices to multiply with stacked_points
        stacked_rots = tf.gather(R, batch_inds)
        # Apply rotations
        stacked_points = tf.reshape(tf.matmul(tf.expand_dims(stacked_points, axis=1), stacked_rots), [-1, 3])
    else:
        raise ValueError('Unknown rotation augmentation : ' + self.augment_rotation)
    # Scale
   
    # Choose random scales for each example
    min_s = augment_scale_min
    max_s = augment_scale_max
    s = tf.random_uniform((num_batches, 3), minval=min_s, maxval=max_s)
    symmetries = []
    
    for i in range(3):
        if augment_symmetries[i]:
            symmetries.append(tf.round(tf.random_uniform((num_batches, 1))) * 2 - 1)
        else:
            symmetries.append(tf.ones([num_batches, 1], dtype=tf.float32))
    s *= tf.concat(symmetries, 1)
    # Create N x 3 vector of scales to multiply with stacked_points
    stacked_scales = tf.gather(s, batch_inds)
    # Apply scales
    stacked_points = stacked_points * stacked_scales
    # Noise
    noise = tf.random_normal(tf.shape(stacked_points), stddev=augment_noise)
    stacked_points = stacked_points + noise
    return stacked_points, s, R



# def tf_augment_input(stacked_points, batch_inds):
#     """
#     Augment inputs with rotation, scale and noise
#     """
#     augment_scale_min = 0.7
#     augment_scale_max = 1.3
#     augment_symmetries = [True, False, False]
#     augment_noise= 0.001
    
#     # Parameter
#     num_batches = batch_inds[-1] + 1

#     ##########
#     # Rotation
#     ##########
#     # if self.augment_rotation == 'none':
#     #     R = tf.eye(3, batch_shape=(num_batches,))
#     # elif self.augment_rotation == 'vertical':
#     #     # Choose a random angle for each element
#     #     theta = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)
#     #     # Rotation matrices
#     #     c, s = tf.cos(theta), tf.sin(theta)
#     #     cs0 = tf.zeros_like(c)
#     #     cs1 = tf.ones_like(c)
#     #     R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
#     #     R = tf.reshape(R, (-1, 3, 3))
#     #     # Create N x 3 x 3 rotation matrices to multiply with stacked_points
#     #     stacked_rots = tf.gather(R, batch_inds)
#     #     # Apply rotations
#     #     stacked_points = tf.reshape(tf.matmul(tf.expand_dims(stacked_points, axis=1), stacked_rots), [-1, 3])
#     # elif self.augment_rotation == 'arbitrarily':
#     #     cs0 = tf.zeros((num_batches,))
#     #     cs1 = tf.ones((num_batches,))
#     #     # x rotation
#     #     thetax = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)
#     #     cx, sx = tf.cos(thetax), tf.sin(thetax)
#     #     Rx = tf.stack([cs1, cs0, cs0, cs0, cx, -sx, cs0, sx, cx], axis=1)
#     #     Rx = tf.reshape(Rx, (-1, 3, 3))
#     #     # y rotation
#     #     thetay = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)
#     #     cy, sy = tf.cos(thetay), tf.sin(thetay)
#     #     Ry = tf.stack([cy, cs0, -sy, cs0, cs1, cs0, sy, cs0, cy], axis=1)
#     #     Ry = tf.reshape(Ry, (-1, 3, 3))
#     #     # z rotation
#     #     thetaz = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)
#     #     cz, sz = tf.cos(thetaz), tf.sin(thetaz)
#     #     Rz = tf.stack([cz, -sz, cs0, sz, cz, cs0, cs0, cs0, cs1], axis=1)
#     #     Rz = tf.reshape(Rz, (-1, 3, 3))
#     #     # whole rotation
#     #     Rxy = tf.matmul(Rx, Ry)
#     #     R = tf.matmul(Rxy, Rz)
#     #     # Create N x 3 x 3 rotation matrices to multiply with stacked_points
#     #     stacked_rots = tf.gather(R, batch_inds)
#     #     # Apply rotations
#     #     stacked_points = tf.reshape(tf.matmul(tf.expand_dims(stacked_points, axis=1), stacked_rots), [-1, 3])
#     # else:
#     #     raise ValueError('Unknown rotation augmentation : ' + self.augment_rotation)
#     stacked_points = tf.reshape(stacked_points,(-1,3))
#     theta = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)
#     # Rotation matrices
#     c, s = tf.cos(theta), tf.sin(theta)
#     cs0 = tf.zeros_like(c)
#     cs1 = tf.ones_like(c)
#     R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
#     R = tf.reshape(R, (-1, 3, 3))
#     # Create N x 3 x 3 rotation matrices to multiply with stacked_points
#     stacked_rots = tf.gather(R, batch_inds)
#     # Apply rotations
#     stacked_points = tf.reshape(tf.matmul(tf.expand_dims(stacked_points, axis=1), stacked_rots), [-1, 3])
#     #######
#     # Scale
#     #######
   
#     # Choose random scales for each example
#     min_s = augment_scale_min
#     max_s = augment_scale_max
#     # if self.augment_scale_anisotropic:
#     #     s = tf.random_uniform((num_batches, 3), minval=min_s, maxval=max_s)
#     # else:
#     #     s = tf.random_uniform((num_batches, 1), minval=min_s, maxval=max_s)
#     s = tf.random_uniform((num_batches, 3), minval=min_s, maxval=max_s)
#     symmetries = []
    
#     for i in range(3):
#         if augment_symmetries[i]:
#             symmetries.append(tf.round(tf.random_uniform((num_batches, 1))) * 2 - 1)
#         else:
#             symmetries.append(tf.ones([num_batches, 1], dtype=tf.float32))
#     s *= tf.concat(symmetries, 1)
#     # Create N x 3 vector of scales to multiply with stacked_points
#     stacked_scales = tf.gather(s, batch_inds)
#     # Apply scales
#     stacked_points = stacked_points * stacked_scales
#     #######
#     # Noise
#     #######
#     noise = tf.random_normal(tf.shape(stacked_points), stddev=augment_noise)
#     stacked_points = stacked_points + noise
#     # print("np.shape(stacked_points) last" )
#     # print(np.shape(stacked_points)) 
#     return stacked_points, s, R



# def tf_get_batch_inds(stacks_len):
#     """
#     Method computing the batch indices of all points, given the batch element sizes (stack lengths). Example:
#     From [3, 2, 5], it would return [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
#     """

#     # Initiate batch inds tensor
#     num_batches = tf.shape(stacks_len)[0]
#     print("num_batches")
#     print(num_batches)
#     num_points = tf.reduce_sum(stacks_len)
#     batch_inds_0 = tf.zeros((num_points,), dtype=tf.int32)

#     # Define body of the while loop
#     def body(batch_i, point_i, b_inds):
#         num_in = stacks_len[batch_i]
#         # num_in = tf.reshape(num_in,())
#         num_before = tf.cond(tf.less(batch_i, 1),
#                                 lambda: tf.zeros((), dtype=tf.int32),
#                                 lambda: tf.reduce_sum(stacks_len[:batch_i]))
#         num_after = tf.cond(tf.less(batch_i, num_batches - 1),
#                             lambda: tf.reduce_sum(stacks_len[batch_i + 1:]),
#                             lambda: tf.zeros((), dtype=tf.int32))

#         # Update current element indices
#         inds_before = tf.zeros((num_before,), dtype=tf.int32)
#         inds_in = tf.fill((num_in,), batch_i)
#         inds_after = tf.zeros((num_after,), dtype=tf.int32)
#         n_inds = tf.concat([inds_before, inds_in, inds_after], axis=0)

#         b_inds += n_inds
#         # Update indices
#         point_i += stacks_len[batch_i]
#         batch_i += 1

#         return batch_i, point_i, b_inds

#     def cond(batch_i, point_i, b_inds):
#         return tf.less(batch_i, tf.shape(stacks_len)[0])

#     _, _, batch_inds = tf.while_loop(cond,
#                                         body,
#                                         loop_vars=[0, 0, batch_inds_0],
#                                         shape_invariants=[tf.TensorShape([]), tf.TensorShape([]),
#                                                         tf.TensorShape([None])])

#     return batch_inds
