from os.path import join, exists, dirname, abspath
from RandLANet import Network
from RandLANet2 import Network2
from RandLANet3 import Network3
from tester_SensatUrban import ModelTester
from helper_ply import read_ply
from tool import ConfigSensatUrban as cfg
from tool import DataProcessing as DP
from tool import Plot
from tool import tf_augment_input
import tensorflow as tf
import numpy as np
import time, pickle, argparse, glob, os, shutil



class SensatUrban:
    def __init__(self):
        self.name = 'SensatUrban'
        root_path = '/hy-tmp/SensatUrban_albert/Dataset'  # path to the dataset
        self.path = join(root_path, self.name)
        self.label_to_names = {0: 'Ground', 1: 'High Vegetation', 2: 'Buildings', 3: 'Walls',
                               4: 'Bridge', 5: 'Parking', 6: 'Rail', 7: 'traffic Roads', 8: 'Street Furniture',
                               9: 'Cars', 10: 'Footpath', 11: 'Bikes', 12: 'Water'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.all_files = np.sort(glob.glob(join(self.path, 'original_block_ply', '*.ply')))
        self.val_file_name = ['birmingham_block_1',
                              'birmingham_block_5',
                              'cambridge_block_10',
                              'cambridge_block_7']
        self.test_file_name = ['birmingham_block_2', 'birmingham_block_8',
                               'cambridge_block_15', 'cambridge_block_22',
                               'cambridge_block_16', 'cambridge_block_27']
        self.use_val = True  # whether use validation set or not

        # initialize
        self.num_per_class = np.zeros(self.num_classes)
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': [], 'test': []}
        self.input_names = {'training': [], 'validation': [], 'test': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)
        for ignore_label in self.ignored_labels:
            self.num_per_class = np.delete(self.num_per_class, ignore_label)
        # data augmentation
                    
        self.enhance_xyz = cfg.enhance_xyz
        self.enhance_color = cfg.enhance_color
        # enhancing pos info
        self.rot_type = cfg.rot_type
        self.augment_scale_min = cfg.augment_scale_min
        self.augment_scale_max = cfg.augment_scale_max
        self.augment_symmetries = cfg.augment_symmetries
        self.augment_noise= cfg.augment_noise

        # dropping color
        self.drop_color = cfg.drop_color
        self.augment_color = cfg.augment_color
        # color Jitter
        self.jitter_color = cfg.jitter_color
        # autocontrast
        self.auto_contrast = cfg.auto_contrast
        self.blend_factor= cfg.blend_factor

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'grid_{:.3f}'.format(sub_grid_size))
        print(enumerate(self.all_files))

        ######                  ######
        ###### Testing     cfg  ######
        ######                  ######
        test = 0

        for i, file_path in enumerate(self.all_files):
            # # loader limit break
            if i>2 and test:
                break
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if cloud_name in self.test_file_name:
                cloud_split = 'test'
            elif cloud_name in self.val_file_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            # compute num_per_class in training set
            if cloud_split == 'training':
                self.num_per_class += DP.get_num_class_from_label(sub_labels, self.num_classes)

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices

        for i, file_path in enumerate(self.all_files):
            # loader limit break
            if i>2 and test:
                break
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # val projection and labels
            if cloud_name in self.val_file_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

            # test projection and labels
            if cloud_name in self.test_file_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.test_proj += [proj_idx]
                self.test_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        else:
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        # Reset possibility
        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]
        
        
        def spatially_regular_gen():
            p_list = []
            c_list = []
            pl_list = []
            pi_list = []
            ci_list = []
            batch_n = 0
            # Generator loop
            for i in range(num_per_epoch):  # num_per_epoch

                # Choose a random cloud
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get points from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                if len(points) < cfg.num_points:
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                n = queried_idx.shape[0]

                queried_idx = DP.shuffle_idx(queried_idx)
                # Collect points and colors
                queried_pc_xyz = points[queried_idx] - pick_point
                queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]

                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

                input_points = queried_pc_xyz.astype(np.float32)
                input_colors = queried_pc_colors
                input_labels = queried_pc_labels
                input_inds = queried_idx.astype(np.int32)
                cloud_ind = cloud_idx

                p_list = []
                c_list = []
                pl_list = []
                pi_list = []
                ci_list = []
                batch_n = 0

                if n > 0:
                    p_list += [input_points]
                    c_list += [input_colors]
                    pl_list += [input_labels]
                    pi_list += [input_inds]
                    ci_list += [cloud_ind]

                batch_n += n

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes
    @staticmethod
    def get_tf_mapping2():

        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []
            # stacks_lengths = tf.reshape(stacks_lengths,(-1,))
            # Get batch indice for each point
            # Augment input points

            ######                  ######
            ###### Data enhancement ######
            ######                  ######
            batch_inds = [cfg.num_points]
            print("batch_xyz")
            print(batch_xyz)
            print("batch_features")
            print(batch_features)
            if cfg.enhance_xyz:
                batch_inds = [cfg.num_points]
                batch_xyz = tf.reshape(batch_xyz,(-1,3))
                batch_xyz, scales, rots = tf_augment_input(batch_xyz, batch_inds, cfg.rot_type,\
                    cfg.augment_scale_min, cfg.augment_scale_max, cfg.augment_symmetries, cfg.augment_noise)
                batch_xyz = tf.reshape(batch_xyz,(cfg.batch_size ,-1,3)) 
            
            if cfg.enhance_color:
                rbg_range = 1  # 255
                if cfg.auto_contrast:
                    # to avoid chromatic drop problems
                    batch_features = tf.reshape(batch_features,(-1,3)) 
                    if np.random.rand() < 0.2:   
                        lo = tf.math.reduce_min(batch_features, axis=0, keepdims=True)
                        hi = tf.math.reduce_max(batch_features, axis=0, keepdims=True)
                        scale = rbg_range / (hi - lo)
                        contrast_feats = (batch_features - lo) * scale
                        batch_features = (1 - cfg.blend_factor) * batch_features + cfg.blend_factor * contrast_feats
                    batch_features = tf.reshape(batch_features,(cfg.batch_size ,-1,3))
                
                if cfg.jitter_color:
                    p=0.95
                    std=0.01
                    batch_features = tf.reshape(batch_features,(-1,3)) 
                    if np.random.rand() < p:
                        noise = np.random.randn(cfg.batch_size*cfg.num_points, 3)
                        noise *= std * rbg_range
                        batch_features = tf.clip_by_value(noise + batch_features, clip_value_min=0, clip_value_max=rbg_range) 
                    batch_features = tf.reshape(batch_features,(cfg.batch_size ,-1,3))

                if cfg.translate_color:
                    trans_range_ratio = 0.95
                    batch_features = tf.reshape(batch_features,(-1,3))
                    if np.random.rand() < trans_range_ratio:
                        tr = np.random.randn(1, 3) * rbg_range * 2 * trans_range_ratio
                        batch_features = tf.clip_by_value(tr + batch_features, clip_value_min=0, clip_value_max=rbg_range) 
                    batch_features = tf.reshape(batch_features,(cfg.batch_size ,-1,3))  
                    
                if cfg.drop_color:
                    # randomly drop colors
                    batch_features = tf.reshape(batch_features,(-1,3)) 
                    num_batches = batch_inds[-1] + 1
                    s = tf.cast(tf.less(tf.random_uniform((num_batches,)), cfg.augment_color), tf.float32)

                    stacked_s = tf.gather(s, batch_inds)
                    batch_features = batch_features * tf.expand_dims(stacked_s, axis=1)
                    batch_features = tf.reshape(batch_features,(cfg.batch_size ,-1,3)) 
                

            # (N, 65536,6)
            
            if cfg.rgb_only:
                batch_features = batch_features
            else:
                batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            print("batch_features after")
            print(batch_features) 
            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:,:tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:,:tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points
            
            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]
            # if cfg.enhance_xyz:
            #     input_list += [scales, rots]
            return input_list
        def tf_map_test_val(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
                
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            # (N, 65536,6)
            if cfg.rgb_only:
                batch_features = batch_features
            else:
                batch_features = tf.concat([batch_xyz, batch_features], axis=-1)

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:,:tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:,:tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points
            
            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]
            return input_list
        return tf_map, tf_map_test_val

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        gen_function_test, _, _ = self.get_batch_gen('test')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        self.batch_test_data = self.test_data.batch(cfg.val_batch_size)

        map_func, map_func_test = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func_test)
        self.batch_test_data = self.batch_test_data.map(map_func=map_func_test)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)
        self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)
        self.test_init_op = iter.make_initializer(self.batch_test_data)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    Mode = FLAGS.mode

    shutil.rmtree('__pycache__') if exists('__pycache__') else None
    if Mode == 'train':
        shutil.rmtree('results') if exists('results') else None
        shutil.rmtree('train_log') if exists('train_log') else None
        for f in os.listdir(dirname(abspath(__file__))):
            if f.startswith('log_'):
                os.remove(f)
    
    dataset = SensatUrban()
    dataset.init_input_pipeline()

    if Mode == 'train':
        # model = Network(dataset, cfg)
        restore_snap = "/hy-tmp/SensatUrban_albert/result/Point-transformer:  sqrt + crossE + xyz: True + color: False/snapshots/snap-(47.14%)-12501"
        # model = Network3(dataset, cfg, None)
        # model.train(dataset)
        model = Network2(dataset, cfg, None)
        model.train(dataset)
    
    elif Mode == 'test':
        cfg.saving = False
        model = Network2(dataset, cfg)
        chosen_snapshot = -1
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
        chosen_folder = logs[-1]
        snap_path = join(chosen_folder, 'snapshots')
        snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(snap_path, 'snap-(51.49%)-{:d}'.format(chosen_step))
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)
        shutil.rmtree('train_log') if exists('train_log') else None

    # elif Mode == 'albert':
    #     with tf.Session() as sess:ate 
    #         sess.run(tf.global_variables_initializer())
    #         sess.run(dataset.test_init_op)
    #        #  sess.run(dataset.train_init_op)
    #         data_list = sess.run(dataset.flat_inputs)
    #         xyz = data_list[0]
    #         print(len(xyz))
    #         print(len(xyz[0, :, :]))
    #         print(xyz[0, :, :])
    #         label = data_list[21]
    #         with open("xyz.pkl", 'wb') as f:
    #                 pickle.dump(xyz, f)
    #         with open("label.pkl", 'wb') as f:
    #             pickle.dump(label, f)
           
    #     print("done")
    else:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            while True:
                data_list = sess.run(dataset.flat_inputs)
                xyz = data_list[0]
                sub_xyz = data_list[1]
                label = data_list[21]
                with open("xyz_flat.pkl", 'wb') as f:
                    pickle.dump(xyz, f)
                with open("label_flat.pkl", 'wb') as f:
                    pickle.dump(label, f)
                # Plot.draw_pc_sem_ins(xyz[0, :, :], label[0, :])
                break
        print("wrong mode!!!!!")
