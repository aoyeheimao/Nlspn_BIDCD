import open3d as o3d
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


def depth_image_to_color_pcd(depth,rgb, scale=1, pose=np.identity(4)):
    K = np.array([[5.1885790117450188e+02 / 2.0, 0.00000000, 3.2558244941119034e+02 / 2.0 - 8.0],
                  [0.00000000, 5.1946961112127485e+02 / 2.0, 2.5373616633400465e+02 / 2.0 - 6.0],
                  [0.00000000, 0.00000000, 1.00000000]])
    K = np.array([[2056.97998046875, 0.00000000, 960.3259887695312],
                       [0.00000000, 2056.580078125, 600.5050048828125],
                       [0.00000000, 0.00000000, 1.00000000]])

    fx = K[0, 0]/3
    cx = K[0, 2]/3
    fy = K[1, 1]/3
    cy =K[1, 2]/3

    depth_array = depth.astype(np.float32)
    rgb =rgb.astype(np.float32)
    imgH, imgW = depth_array.shape

    camera_param = o3d.camera.PinholeCameraIntrinsic(width=imgW, height=imgH, fx=fx, fy=fy, cx=cx, cy=cy)
    o3drgb = o3d.geometry.Image(rgb/256)
    o3ddep = o3d.geometry.Image(depth_array)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3drgb, depth=o3ddep)
    scene_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic=camera_param)#.voxel_down_sample(voxel_size=1)  # , extrinsic=extrinsic)
    # scene_o3d = scene_o3d.paint_uniform_color([1, 0, 0])
    return scene_o3d

    # # cleaned_depth, gt_depth, masks_workspace, raw_dept, rgb, gray = self[item]
    #
    # # rgb = gray  # 模拟灰度图像
    #
    # # depth = np.array(gt_depth)
    # K = np.array([[5.1885790117450188e+02 / 2.0, 0.00000000, 3.2558244941119034e+02 / 2.0 - 8.0],
    #               [0.00000000, 5.1946961112127485e+02 / 2.0, 2.5373616633400465e+02 / 2.0 - 6.0],
    #               [0.00000000, 0.00000000, 1.00000000]])
    # u = range(0, rgb.shape[1])
    # v = range(0, rgb.shape[0])
    #
    # u, v = np.meshgrid(u, v)
    # u = u.astype(float)
    # v = v.astype(float)
    #
    # Z = depth.astype(float) / scale
    # X = (u - K[0, 2]) * Z / K[0, 0]
    # Y = (v - K[1, 2]) * Z / K[1, 1]
    #
    # X = np.ravel(X)
    # Y = np.ravel(Y)
    # Z = np.ravel(Z)
    #
    # valid = Z > 0
    #
    # X = X[valid]
    # Y = Y[valid]
    # Z = Z[valid]
    #
    # position = np.vstack((X, Y, Z, np.ones(len(X))))
    # position = np.dot(pose, position)
    #
    # R = np.ravel(rgb[:, :, 0])[valid]
    # G = np.ravel(rgb[:, :, 1])[valid]
    # B = np.ravel(rgb[:, :, 2])[valid]
    #
    # points = np.transpose(np.vstack((position[0:3, :], R, G, B)))  # .tolist()
    #
    # colors = points[:, 3:6] / 255
    # points = points[:, :3]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    #
    # return pcd
def visualize_BIDCD():
    from src.data.rgbddataset import RGBDDATASET
    from src.data.si import SI
    from src.config import args as args_config
    import argparse
    import torch
    def check_args(args):
        new_args = args
        return new_args

    parser = argparse.ArgumentParser(description='NLSPN')

    args = parser.parse_args()
    args.augment = None
    args.depth_type = None
    args.depth_type = 'scan'

    args_main = check_args(args)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ':',  getattr(args_main, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')
    for i in range(0,100,10):
        dataset = SI(args_main,'train')[i]
        # dataset = RGBDDATASET(args_main,'test')[2]
        print(dataset['rgb'].shape)
        print('rgb_max:', dataset['rgb'].max(), 'rgb_min:', dataset['rgb'].min(), 'mean:', dataset['rgb'].mean())
        from matplotlib import pyplot as plt

        plt.imshow(dataset['dep'][0, :, :])
        plt.show()
        plt.imshow(dataset['gt'][0, :, :])
        plt.show()
        plt.imshow(np.transpose(dataset['rgb'], (1, 2, 0)))
        plt.show()



def visualize_png():
    dir_path = r'C:\Users\HJH\Desktop\train_log\802release_one_layer_endecoder\val\epoch1105\00000009'
    img_gt = cv2.imread(os.path.join(dir_path,'06_gt.png'), 2)
    img_pre = cv2.imread(os.path.join(dir_path,'05_pred_final.png'), 2)
    img_scan = cv2.imread(os.path.join(dir_path,'02_dep.png'), 2)
    img_rgb = cv2.imread(os.path.join(dir_path,'01_rgb.png'))

    img_init_pre = cv2.imread(os.path.join(dir_path,'03_pred_init.png'), 2)
    print(img_gt.shape)
    plt.imshow(img_gt)
    plt.show()

    pcd = depth_image_to_color_pcd(img_scan, img_rgb)
    o3d.visualization.draw_geometries([pcd], 'pcd')

    pcd = depth_image_to_color_pcd(img_pre, img_rgb)
    o3d.visualization.draw_geometries([pcd], 'pcd')

    pcd = depth_image_to_color_pcd(img_gt, img_rgb)
    o3d.visualization.draw_geometries([pcd], 'pcd')

    pcd = depth_image_to_color_pcd(img_init_pre, img_rgb)
    o3d.visualization.draw_geometries([pcd], 'pcd')


if __name__ =='__main__':
    # visualize_png()

    # visualize_BIDCD()
    # exit()
    dir_path = r'Z:\mnt\sdd\data\zhoutianyi\docker_data4\Nlspn_BIDCD\experiments\220622_124213_render_data_train_norefine\val\epoch0454\00000035'
    data_f = np.load(os.path.join(dir_path,'array_result_save.npz'))

    rgb = data_f['rgb']
    dep = data_f['dep']
    pred = data_f['pred']
    pred_gray = data_f['pred_gray']
    gt = data_f['gt']

    pcd = depth_image_to_color_pcd(dep, rgb)
    o3d.visualization.draw_geometries([pcd], 'pcd')
    o3d.io.write_point_cloud('dep.ply',pcd)

    pred[gt == 0] = 0
    pred_pcd = depth_image_to_color_pcd(pred, rgb)
    cl, ind = pred_pcd.remove_statistical_outlier(nb_neighbors=30,
                                                  std_ratio=2)
    pred_pcd = pred_pcd.select_by_index(ind)
    o3d.visualization.draw_geometries([pred_pcd], 'pred_pcd')
    o3d.io.write_point_cloud('pred_pcd.ply', pred_pcd)

    pred_pcd.paint_uniform_color([1, 0, 0])
    pcd = depth_image_to_color_pcd(gt, rgb)
    o3d.visualization.draw_geometries([pred_pcd, pcd], 'pcd')
    o3d.io.write_point_cloud('gt.ply', pcd)

    feat_init = data_f['feat_init']
    pcd = depth_image_to_color_pcd(feat_init, rgb)
    o3d.visualization.draw_geometries([pcd], 'pcd')

