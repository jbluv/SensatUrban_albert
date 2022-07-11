from os.path import join, exists, dirname
from sklearn.neighbors import KDTree
from tool import DataProcessing as DP
from helper_ply import write_ply
import numpy as np
import os, pickle, argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='the number of GPUs to use [default: 0]')
    FLAGS = parser.parse_args()
    dataset_name = 'SensatUrban'
    dataset_path = FLAGS.dataset_path
    preparation_types = ['grid']  # Grid sampling & Random sampling
    grid_size = 0.2
    random_sample_ratio = 10
    # temp_files = np.sort([join(dataset_path, 'original_block_ply', i) for i in os.listdir(join(dataset_path, 'original_block_ply'))])
    temp_files = np.sort([join(dataset_path, 'test', i) for i in os.listdir(join(dataset_path, 'test'))])

    files = np.sort(np.hstack((temp_files)))
    test_files_name = ["birmingham_block_2","birmingham_block_8","cambridge_block_22",
                        "cambridge_block_27","cambridge_block_16","cambridge_block_15"]
    for sample_type in preparation_types:
        for pc_path in files:
            cloud_name = pc_path.split('/')[-1][:-4]
            print('start to process:', cloud_name)

            # create output directory
            out_folder = join(dirname(dataset_path), sample_type + '_{:.3f}'.format(grid_size))
            os.makedirs(out_folder) if not exists(out_folder) else None

            if pc_path in temp_files:
                if cloud_name in test_files_name:
                    xyz, rgb = DP.read_ply_data(pc_path, with_rgb=True, with_label=False)
                    
                    import pickle
                    with open("xyz.pkl", 'wb') as f:
                        pickle.dump(xyz, f)
                    print(len(xyz))
