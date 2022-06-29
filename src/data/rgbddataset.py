from torch.utils.data import Dataset
import numpy as np
import glob
import cv2
import os
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from . import BaseDataset
import os
import warnings
import numpy as np
import json
import h5py
from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class RGBDDATASET(BaseDataset):
    """Dataset wrapping data and target tensors.
    Each sample will be retrieved by indexing both tensors along the first
    dimension.
    Arguments:
        folder_name: the path in project that store RGB-D data.
        dir_path: only use if the dataset is not in project root folder.
    """

    def __init__(self, args, mode):
        super(RGBDDATASET, self).__init__(args, mode)
        self.dir = '..'  # ..
        self.fold = 'data/*/*'  # data/*/*
        self.cleaned_depth = sorted(glob.glob(os.path.join(self.dir, self.fold, 'cleaned_depth/*.npz')))
        self.gt_depth = sorted(glob.glob(os.path.join(self.dir, self.fold, 'gt_depth/*.npz')))
        self.masks_workspace = sorted(glob.glob(os.path.join(self.dir, self.fold, 'masks_workspace/*.npz')))
        self.raw_dept = sorted(glob.glob(os.path.join(self.dir, self.fold, 'raw_depth/*.npz')))
        self.rgb = sorted(glob.glob(os.path.join(self.dir, self.fold, 'rgb/*.png')))

        # 额外的参数, K是NYU数据集借来的
        # self.K = torch.Tensor([5.1885790117450188e+02 / 2.0,
        #     5.1946961112127485e+02 / 2.0,
        #     3.2558244941119034e+02 / 2.0 - 8.0,
        #     2.5373616633400465e+02 / 2.0 - 6.0
        # ])
        self.K = torch.Tensor([5.1885790117450188e+02 / 2.0,
                               5.1946961112127485e+02 / 2.0,
                               3.2558244941119034e+02 / 2.0 - 8.0,
                               2.5373616633400465e+02 / 2.0 - 6.0
                               ]) / 2
        # 照片大小 需要定义
        height, width = (240, 428)  # (720, 1280)
        crop_size = (216, 384)  # (228, 304)

        self.height = height
        self.width = width
        self.crop_size = crop_size
        self.augment = self.args.augment
        self.depth_type = self.args.depth_type

        if self.mode == 'train':
            self.gt_depth = self.gt_depth[:round(len(self.gt_depth) * 0.8)]
            self.raw_dept = self.raw_dept[:round(len(self.raw_dept) * 0.8)]
            self.rgb = self.rgb[:round(len(self.rgb) * 0.8)]

        else:
            self.gt_depth = self.gt_depth[round(len(self.gt_depth) * 0.8):len(self.gt_depth)]
            self.raw_dept = self.raw_dept[round(len(self.raw_dept) * 0.8):len(self.raw_dept)]
            self.rgb = self.rgb[round(len(self.rgb) * 0.8):len(self.rgb)]

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, idx):
        # torch.from_numpy(np.load(self.cleaned_depth[idx])['arr_0']), \
        # torch.from_numpy(np.load(self.gt_depth[idx])['arr_0']), \
        # torch.from_numpy(np.load(self.masks_workspace[idx])['arr_0']), \
        # torch.from_numpy(np.load(self.raw_dept[idx])['arr_0']), \
        # torch.from_numpy(np.array(cv2.imread(self.rgb[idx]))), \
        # torch.from_numpy(np.array(cv2.cvtColor(cv2.imread(self.rgb[idx], 0), cv2.COLOR_GRAY2RGB)))

        # rgb = np.array(cv2.imread(self.rgb[idx])) # rgb图

        # 在loss里面已经存在了一个阈值mask不需要额外添加了
        masks_workspace = np.load(self.masks_workspace[idx])['arr_0'][::3, ::3]  # mask
        # masks_workspace_01 = np.ones_like(masks_workspace)
        # masks_workspace_01[masks_workspace == False] = 0  # 创造一个01mask传入模型，在loss阶段覆盖上去防止被反向传播

        rgb = np.array(cv2.cvtColor(cv2.imread(self.rgb[idx], 0), cv2.COLOR_GRAY2RGB))[::3, ::3, :]  # rgb模拟的三通道灰度图
        rgb[masks_workspace == False] = 0
        dep = np.load(self.gt_depth[idx])['arr_0'][::3, ::3] / 100  # 恢复的目标深度
        dep[masks_workspace == False] = 0
        scan_dep = np.load(self.raw_dept[idx])['arr_0'][::3, ::3] / 100  # 工件扫描深度图
        scan_dep[masks_workspace == False] = 0

        # print('rgb_maxmin:', rgb.max(),rgb.min(), rgb.shape)
        # print('dep_maxmin:', dep.max(),dep.min(), dep.shape)
        # print('scan_dep_maxmin:', scan_dep.max(),scan_dep.min(), scan_dep.shape)
        # mask_dep = np.load(self.masks_workspace[idx])  # 用于计算loss的区域，在后续版本可以加如

        # 从array转换成pil image，其他loader是从h5
        # print(dep, rgb)
        rgb = Image.fromarray(rgb, mode='RGB')
        dep = Image.fromarray(dep.astype('float32'), mode='F')
        scan_dep = Image.fromarray(scan_dep.astype('float32'), mode='F')
        # print(dep.size,rgb.size)

        if self.augment and self.mode == 'train':
            _scale = np.random.uniform(1.0, 1.5)
            scale = np.int(self.height * _scale)
            degree = np.random.uniform(-90.0, 90.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)
                scan_dep = TF.hflip(scan_dep)

            rgb = TF.rotate(rgb, angle=degree,
                            resample=Image.NEAREST)  # Argument resample is deprecated and will be removed since v0.10.0. Used interpolation instead
            #  UserWarning: Argument interpolation should be of type InterpolationMode instead of int. Please, use InterpolationMode enum.
            dep = TF.rotate(dep, angle=degree, resample=Image.NEAREST, fill=(0,))
            scan_dep = TF.rotate(scan_dep, angle=degree, resample=Image.NEAREST,
                                 fill=(0,))  # bug appear when using torchvision==0.5.0, by adding fill=(0,) fix it.

            t_rgb = T.Compose([
                T.Resize(scale),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_dep = T.Compose([
                T.Resize(scale),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])
            # t_scan_dep = T.Compose([ # 不能分开随机
            #     T.Resize(scale),
            #     T.CenterCrop(self.crop_size),
            #     self.ToNumpy(),
            #     T.ToTensor()
            # ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)
            scan_dep = t_dep(scan_dep)

            dep = dep / _scale
            scan_dep = scan_dep / _scale

            K = self.K.clone()
            K[0] = K[0] * _scale
            K[1] = K[1] * _scale
        else:
            t_rgb = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])
            # t_scan_dep = T.Compose([
            #     T.Resize(self.height),
            #     T.CenterCrop(self.crop_size),
            #     self.ToNumpy(),
            #     T.ToTensor()
            # ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)
            scan_dep = t_dep(scan_dep)

            K = self.K.clone()

        if self.depth_type == 'generate':
            dep_sp = self.get_sparse_depth(dep, self.args.num_sample)  # 稀疏深度的采样方式，后续可以通过raw depth替换。
        elif self.depth_type == 'scan':
            dep_sp = scan_dep

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K}

        return output

    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel * height * width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp

    def show_data(self, item):

        cleaned_depth, gt_depth, masks_workspace, raw_dept, rgb, gray = self[item]

        plt.subplot(231)
        plt.imshow(cleaned_depth), plt.axis('off'), plt.title('cleaned_depth')
        plt.subplot(232)
        plt.imshow(gt_depth), plt.axis('off'), plt.title('gt_depth')
        plt.subplot(233)
        plt.imshow(masks_workspace), plt.axis('off'), plt.title('masks_workspace')
        plt.subplot(234)
        plt.imshow(raw_dept), plt.axis('off'), plt.title('raw_dept')
        plt.subplot(235)
        plt.imshow(np.array(rgb)), plt.axis('off'), plt.axis('off'), plt.title('rgb')
        plt.subplot(236)
        plt.imshow(np.array(gray)), plt.axis('off'), plt.axis('off'), plt.title('gray')

        plt.show()

    def depth_image_to_color_pcd(self, item, scale=1, pose=np.identity(4)):

        cleaned_depth, gt_depth, masks_workspace, raw_dept, rgb, gray = self[item]

        rgb = gray  # 模拟灰度图像

        depth = np.array(gt_depth)
        K = np.array([[1.01015161e+03, 0.00000000, 6.72047180e+02],
                      [0.00000000, 1.01018524e+03, 4.86441254e+02],
                      [0.00000000, 0.00000000, 1.00000000]])
        u = range(0, rgb.shape[1])
        v = range(0, rgb.shape[0])

        u, v = np.meshgrid(u, v)
        u = u.astype(float)
        v = v.astype(float)

        Z = depth.astype(float) / scale
        X = (u - K[0, 2]) * Z / K[0, 0]
        Y = (v - K[1, 2]) * Z / K[1, 1]

        X = np.ravel(X)
        Y = np.ravel(Y)
        Z = np.ravel(Z)

        valid = Z > 0

        X = X[valid]
        Y = Y[valid]
        Z = Z[valid]

        position = np.vstack((X, Y, Z, np.ones(len(X))))
        position = np.dot(pose, position)

        R = np.ravel(rgb[:, :, 0])[valid]
        G = np.ravel(rgb[:, :, 1])[valid]
        B = np.ravel(rgb[:, :, 2])[valid]

        points = np.transpose(np.vstack((position[0:3, :], R, G, B)))  # .tolist()

        colors = points[:, 3:6] / 255
        points = points[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

# #
# # #
# # # test
# import argparse
# def check_args(args):
#     if args.batch_size < args.num_gpus:
#         print("batch_size changed : {} -> {}".format(args.batch_size,
#                                                      args.num_gpus))
#         args.batch_size = args.num_gpus

#     new_args = args
#     if args.pretrain is not None:
#         assert os.path.exists(args.pretrain), \
#             "file not found: {}".format(args.pretrain)

#         if args.resume:
#             checkpoint = torch.load(args.pretrain)

#             new_args = checkpoint['args']
#             new_args.test_only = args.test_only
#             new_args.pretrain = args.pretrain
#             new_args.dir_data = args.dir_data
#             new_args.resume = args.resume

#     return new_args
# parser = argparse.ArgumentParser(description='NLSPN')


# # Dataset
# parser.add_argument('--dir_data',
#                     type=str,
#                     default='/HDD/dataset/NYUDepthV2_HDF5',
#                     # default='/HDD/dataset/KITTIDepthCompletion',
#                     help='path to dataset')
# parser.add_argument('--data_name',
#                     type=str,
#                     default='NYU',
#                     # default='KITTIDC',
#                     choices=('NYU', 'KITTIDC', 'RGBDDATASET','DFX802'),
#                     help='dataset name')
# parser.add_argument('--split_json',
#                     type=str,
#                     default='../data_json/nyu.json',
#                     # default='../data_json/kitti_dc.json',
#                     help='path to json file')
# parser.add_argument('--patch_height',
#                     type=int,
#                     default=228,
#                     # default=240,
#                     help='height of a patch to crop')
# parser.add_argument('--patch_width',
#                     type=int,
#                     default=304,
#                     # default=1216,
#                     help='width of a patch to crop')
# parser.add_argument('--top_crop',
#                     type=int,
#                     default=0,
#                     # default=100,
#                     help='top crop size for KITTI dataset')
# parser.add_argument('--depth_type',
#                     type=str,
#                     default='scan',
#                     choices=('scan', 'generate'),
#                     help='spase depth type')  # 生成depth的方法，原本是使用随机采样，替换成了扫描出来的数据集


# # Hardware
# parser.add_argument('--seed',
#                     type=int,
#                     default=7240,
#                     help='random seed point')
# parser.add_argument('--gpus',
#                     type=str,
#                     default="0,1,2,3",
#                     help='visible GPUs')
# parser.add_argument('--port',
#                     type=str,
#                     default='29500',
#                     help='multiprocessing port')
# parser.add_argument('--num_threads',
#                     type=int,
#                     default=1,
#                     help='number of threads')
# parser.add_argument('--no_multiprocessing',
#                     action='store_true',
#                     default=False,
#                     help='do not use multiprocessing')


# # Network
# parser.add_argument('--model_name',
#                     type=str,
#                     default='NLSPN',
#                     choices=('NLSPN',),
#                     help='model name')
# parser.add_argument('--network',
#                     type=str,
#                     default='resnet34',
#                     choices=('resnet18', 'resnet34'),
#                     help='network name')
# parser.add_argument('--from_scratch',
#                     action='store_true',
#                     default=False,
#                     help='train from scratch')
# parser.add_argument('--prop_time',
#                     type=int,
#                     default=18,
#                     help='number of propagation')
# parser.add_argument('--prop_kernel',
#                     type=int,
#                     default=3,
#                     help='propagation kernel size')
# parser.add_argument('--preserve_input',
#                     action='store_true',
#                     default=False,
#                     help='preserve input points by replacement')
# parser.add_argument('--affinity',
#                     type=str,
#                     default='TGASS',
#                     choices=('AS', 'ASS', 'TC', 'TGASS'),
#                     help='affinity type (dynamic pos-neg, dynamic pos, '
#                          'static pos-neg, static pos, none')
# parser.add_argument('--affinity_gamma',
#                     type=float,
#                     default=0.5,
#                     help='affinity gamma initial multiplier '
#                          '(gamma = affinity_gamma * number of neighbors')
# parser.add_argument('--conf_prop',
#                     action='store_true',
#                     default=True,
#                     help='confidence for propagation')
# parser.add_argument('--no_conf',
#                     action='store_false',
#                     dest='conf_prop',
#                     help='no confidence for propagation')
# parser.add_argument('--legacy',
#                     action='store_true',
#                     default=False,
#                     help='legacy code support for pre-trained models')


# # Training
# parser.add_argument('--loss',
#                     type=str,
#                     default='1.0*L1+1.0*L2',
#                     help='loss function configuration')
# parser.add_argument('--opt_level',
#                     type=str,
#                     default='O0',
#                     choices=('O0', 'O1', 'O2', 'O3'))
# parser.add_argument('--pretrain',
#                     type=str,
#                     default=None,
#                     help='ckpt path')
# parser.add_argument('--resume',
#                     action='store_true',
#                     help='resume training')
# parser.add_argument('--test_only',
#                     action='store_true',
#                     help='test only flag')
# parser.add_argument('--epochs',
#                     type=int,
#                     default=20,
#                     help='number of epochs to train')
# parser.add_argument('--batch_size',
#                     type=int,
#                     default=12,
#                     help='input batch size for training')
# parser.add_argument('--max_depth',
#                     type=float,
#                     default=1000.0,
#                     # default=90.0,
#                     help='maximum depth')
# parser.add_argument('--augment',
#                     type=bool,
#                     default=True,
#                     help='data augmentation')
# parser.add_argument('--no_augment',
#                     action='store_false',
#                     dest='augment',
#                     help='no augmentation')
# parser.add_argument('--num_sample',
#                     type=int,
#                     default=100,
#                     # default=0,
#                     help='number of sparse samples')
# parser.add_argument('--test_crop',
#                     action='store_true',
#                     default=False,
#                     help='crop for test')


# # Summary
# parser.add_argument('--num_summary',
#                     type=int,
#                     default=4,
#                     help='maximum number of summary images to save')


# # Optimizer
# parser.add_argument('--lr',
#                     type=float,
#                     default=0.001,
#                     help='learning rate')
# parser.add_argument('--decay',
#                     type=str,
#                     default='10,15,20',
#                     help='learning rate decay schedule')
# parser.add_argument('--gamma',
#                     type=str,
#                     default='1.0,0.2,0.04',
#                     help='learning rate multiplicative factors')
# parser.add_argument('--optimizer',
#                     default='ADAM',
#                     choices=('SGD', 'ADAM', 'RMSprop'),
#                     help='optimizer to use (SGD | ADAM | RMSprop)')
# parser.add_argument('--momentum',
#                     type=float,
#                     default=0.9,
#                     help='SGD momentum')
# parser.add_argument('--betas',
#                     type=tuple,
#                     default=(0.9, 0.999),
#                     help='ADAM beta')
# parser.add_argument('--epsilon',
#                     type=float,
#                     default=1e-8,
#                     help='ADAM epsilon for numerical stability')
# parser.add_argument('--weight_decay',
#                     type=float,
#                     default=0.0,
#                     help='weight decay')
# parser.add_argument('--warm_up',
#                     action='store_true',
#                     default=True,
#                     help='do lr warm up during the 1st epoch')
# parser.add_argument('--no_warm_up',
#                     action='store_false',
#                     dest='warm_up',
#                     help='no lr warm up')

# # Logs
# parser.add_argument('--save',
#                     type=str,
#                     default='trial',
#                     help='file name to save')
# parser.add_argument('--save_full',
#                     action='store_true',
#                     default=False,
#                     help='save optimizer, scheduler and amp in '
#                          'checkpoints (large memory)')
# parser.add_argument('--save_image',
#                     action='store_true',
#                     default=False,
#                     help='save images for test')
# parser.add_argument('--save_result_only',
#                     action='store_true',
#                     default=False,
#                     help='save result images only with submission format')


# args = parser.parse_args()

# args.num_gpus = len(args.gpus.split(','))
# args_main = check_args(args)

# print('\n\n=== Arguments ===')
# cnt = 0
# for key in sorted(vars(args_main)):
#     print(key, ':',  getattr(args_main, key), end='  |  ')
#     cnt += 1
#     if (cnt + 1) % 5 == 0:
#         print('')
# print('\n')
# for i in range(20,50):
#     dataset = RGBDDATASET(args_main,'train')[i]
#     # dataset = RGBDDATASET(args_main,'test')[2]
#     print(dataset['rgb'].shape)
#     print('rgb_max:', dataset['rgb'].max(), 'rgb_min:', dataset['rgb'].min(), 'mean:', dataset['rgb'].mean())
#     from matplotlib import pyplot as plt


#     plt.imshow(dataset['dep'][0, :, :])
#     plt.show()
#     plt.imshow(dataset['gt'][0, :, :])
#     plt.show()
#     plt.imshow(np.transpose(dataset['rgb'], (1, 2, 0)))
#     plt.show()
