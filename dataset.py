
import numpy as np 
import os 
import jittor as jt 
from jittor.dataset.dataset import Dataset, dataset_root 
import h5py 

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNxKX6 array, original batch of point clouds
        Return:
          [BxNxKX3, BXNXKX3] array, rotated batch of point clouds
    """
    rotated_points = np.zeros((batch_data.shape[0],batch_data.shape[1],batch_data.shape[2],3), dtype=np.float32)
    rotated_pts = np.zeros((batch_data.shape[0],batch_data.shape[1],batch_data.shape[2],3), dtype=np.float32)
    rotated_ns = np.zeros((batch_data.shape[0],batch_data.shape[1],batch_data.shape[2],3), dtype=np.float32)
    
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, : , : , :3]
        rotated_points[k, :,:,:3] = np.dot(shape_pc, rotation_matrix)
        shape_pc_pt = batch_data[k, : , : , 3:6]
        rotated_pts[k, :, :, :3] = np.dot(shape_pc_pt , rotation_matrix)
        shape_pc_n = batch_data[k, : , : , 9:12]
        rotated_ns[k, :, :, :3] = np.dot(shape_pc_n , rotation_matrix)
        del shape_pc
        del shape_pc_pt
        del shape_pc_n
        del rotation_matrix
    return rotated_points, rotated_pts, rotated_ns


class SVNetTrainDataSet(Dataset):
    def __init__(self , train_h5_file , num_points_ , num_k_ , num_class_):
        super().__init__()

        self.num_points = num_points_
        self.num_k = num_k_
        self.num_class = num_class_
        cur_points , cur_labels = h5py.load_h5(train_h5_file)
        self.train_feature_r = cur_points.reshape(-1 , self.num_points , self.num_k , 33)
        self.train_label_r = cur_labels.reshape(-1, self.num_points , self.num_class)

    def __getitem__(self , index):

        pnts = self.train_feature_r[index, :, : , :]
        labs = self.train_label_r[index, :, : , :]

        train_points_rotate , train_feature_rotate, train_norm_rotate = rotate_point_cloud(pnts[:,:,:,:12])
        train_feature_rotate_total = np.concate([train_points_rotate, pnts[:,:,:,6:9] , train_norm_rotate, pnts[:,:,:,12:]], axis = 3)

        pnts_cuda = jt.array(train_feature_rotate_total)
        labs_cuda = jt.array(labs)

        return pnts_cuda , labs_cuda

    def __len__(self):

        return self.train_feature_r.shape[0]


class SVNetTestDataSet(Dataset):
    def __init__(self , test_h5_file , num_points_ , num_k_ , num_class_):
        super().__init__()

        self.num_points = num_points_
        self.num_k = num_k_
        self.num_class = num_class_
        cur_points , cur_labels = h5py.load_h5(train_h5_file)
        self.train_feature_r = cur_points.reshape(-1 , self.num_points , self.num_k , 33)
        self.train_label_r = cur_labels.reshape(-1, self.num_points , self.num_class)

    def __getitem__(self , index):

        pnts = self.train_feature_r[index, :, : , :]
        labs = self.train_label_r[index, :, : , :]

        # train_points_rotate , train_feature_rotate, train_norm_rotate = rotate_point_cloud(pnts[:,:,:,:12])
        # train_feature_rotate_total = np.concate([train_points_rotate, pnts[:,:,:,6:9] , train_norm_rotate, pnts[:,:,:,12:]], axis = 3)

        pnts_cuda = jt.array(pnts)
        labs_cuda = jt.array(labs)

        return pnts_cuda , labs_cuda

    def __len__(self):

        return self.train_feature_r.shape[0]

