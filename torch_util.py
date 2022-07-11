import tensorflow as tf
# This file is a test file, its not official.
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return tf.reduce_sum((src[:, :, None] - dst[:, None]) ** 2, axis=-1)

def torch_gather(x,gather_axis, indices):

    all_indices = tf.where(tf.fill(indices.shape, True))
    gather_locations = tf.reshape(indices, [indices.shape.num_elements()])

    gather_indices = []
    for axis in range(len(indices.shape)):
        if axis == gather_axis:
            gather_indices.append(tf.cast(gather_locations, dtype=tf.int64))
        else:
            gather_indices.append(tf.cast(all_indices[:, axis], dtype=tf.int64))

    gather_indices = tf.stack(gather_indices, axis=-1)
    gathered = tf.gather_nd(x, gather_indices)
    reshaped = tf.reshape(gathered, indices.shape)
    return reshaped

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch_gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

import numpy as np
import tensorflow as tf


def index_sieve(features, indices, fix_dim=2):
    features_sieve = tf.gather(features[:, :,...], indices[:, :,...], axis=-2, batch_dims=fix_dim)
#     features_sieve = tf.concat([tf.expand_dims(tf.gather(features[:, i,...], indices[:, i,...], axis=-2, batch_dims=1), axis=1) for i in range(features.shape[1])], axis=1)
    return features_sieve

def rotate_point_cloud(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data




# dists = square_distance(xyz, xyz)
# knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
# knn_xyz = index_points(xyz, knn_idx)

# pre = features
# x = self.fc1(features)
# q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

# pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

# attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
# attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

# res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
# res = self.fc2(res) + pre