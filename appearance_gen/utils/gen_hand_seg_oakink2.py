import pickle

import os
import open3d as o3d
import numpy as np

from PIL import Image, ImageDraw

from manotorch.manolayer import ManoLayer

import torch
import cv2
import sys
import math
import glob
from tqdm import tqdm
import json

import argparse

import argparse
from argparse import ArgumentParser

COLORS = [
    [0, 0, 0],
    [92, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
]

CLASS_PROTOCAL = [
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [0, 0, 0],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
    #   [0, 0, 0]
  ]

def convert_gray_to_color(gray_image, color_map=CLASS_PROTOCAL[21:]):
    color_image_c1 = []
    color_image_c2 = []
    color_image_c3 = []
    h, w = gray_image.shape
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            color = gray_image[i, j]
            color_image_c1.append(color_map[color][0])
            color_image_c2.append(color_map[color][1])
            color_image_c3.append(color_map[color][2])
    return np.stack([np.asarray(color_image_c1, dtype=np.uint8),
                    np.asarray(color_image_c2, dtype=np.uint8),
                    np.asarray(color_image_c3, dtype=np.uint8)], axis=-1).reshape(h, w, 3)
    
def showHandJoints(imgInOrg, gtIn, gtInDepth=None, is_right=False, filename=None):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param gtIn3D: 3D ground truth annotation
    :param filename: dump image name
    :return:
    '''

    # imgIn = np.zeros_like(imgInOrg)
    imgIn = np.copy(imgInOrg)

    # Set color for each finger
    if is_right:
        joint_color_code = [[255, 0, 0], 
                            [255, 85, 85],
                            [255, 170, 170],
                            [255, 255, 255],
                            [200, 200, 200],
                            [150, 150, 150]]
    else:
        joint_color_code = [[139, 53, 255],
                            [0, 56, 255],
                            [43, 140, 237],
                            [37, 168, 36],
                            [147, 147, 0],
                            [70, 17, 145]]

    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]]

    # Judge the order of limbs
    reverse = False
    if gtInDepth is not None:
        if gtInDepth[0] < gtInDepth[-1]: 
            # Thumbs on the top layer because the z value is smaller, draw finally
            reverse = True

    PYTHON_VERSION = sys.version_info[0]

    gtIn = np.round(gtIn).astype(np.int32)

    try:
        if gtIn.shape[0]==1:
            imgIn = cv2.circle(imgIn, center=(gtIn[0][0], gtIn[0][1]), radius=3, color=joint_color_code[0],
                                thickness=-1)
        else:

            for joint_num in range(gtIn.shape[0]):

                color_code_num = (joint_num // 4)
                if joint_num in [0, 4, 8, 12, 16]:
                    if PYTHON_VERSION == 3:
                        joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                    else:
                        joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                    cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)
                else:
                    if PYTHON_VERSION == 3:
                        joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                    else:
                        joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                    cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)

            range_indices = range(len(limbs)) if not reverse else range(len(limbs) - 1, -1, -1)
            for limb_num in range_indices:
                x1 = gtIn[limbs[limb_num][0], 1]
                y1 = gtIn[limbs[limb_num][0], 0]
                x2 = gtIn[limbs[limb_num][1], 1]
                y2 = gtIn[limbs[limb_num][1], 0]
                length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if length < 150 and length > 5:
                    deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                    if limb_num < 4:
                        polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                                (int(length / 2), 4),
                                                int(deg),
                                                0, 360, 1)
                    else:
                        polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                                (int(length / 2), 2),
                                                int(deg),
                                                0, 360, 1)
                    color_code_num = limb_num // 4
                    if PYTHON_VERSION == 3:
                        limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
                    else:
                        limb_color = map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num])

                    cv2.fillConvexPoly(imgIn, polygon, color=limb_color)
    except Exception as e:
        print('Error in drawing hand joints. Passing the original image')

    return imgIn

def get_face_rh(mano_layer_rh):
    return mano_layer_rh.get_mano_closed_faces().cpu().numpy()

def get_face_lh(mano_layer_lh):
    _close_faces = torch.Tensor(
        [
            [92, 38, 122],
            [234, 92, 122],
            [239, 234, 122],
            [279, 239, 122],
            [215, 279, 122],
            [215, 122, 118],
            [215, 118, 117],
            [215, 117, 119],
            [215, 119, 120],
            [215, 120, 108],
            [215, 108, 79],
            [215, 79, 78],
            [215, 78, 121],
            [214, 215, 121],
        ]
    )
    _th_closed_faces = torch.cat([mano_layer_lh.th_faces.clone().detach().cpu(), _close_faces[:, [2, 1, 0]].long()])
    hand_faces_lh = _th_closed_faces.cpu().numpy()
    return hand_faces_lh

def load_hand_data(mano_param, is_right, extrinsic, intrinsic, device=torch.device("cpu")):
    if is_right:
        mano_layer = ManoLayer(
            mano_assets_root="asset/mano_v1_2",
            rot_mode="quat",
            side="right",
            center_idx=0,
            use_pca=False,
            flat_hand_mean=True,
        )
        hand_faces = get_face_rh(mano_layer)
        prefix = "rh__"
    else:
        mano_layer = ManoLayer(
            mano_assets_root="asset/mano_v1_2",
            rot_mode="quat",
            side="left",
            center_idx=0,
            use_pca=False,
            flat_hand_mean=True,
        )
        hand_faces = get_face_lh(mano_layer)
        prefix = "lh__"
        
    mano_out = mano_layer(pose_coeffs=mano_param[prefix+'pose_coeffs'], betas=mano_param[prefix+'betas'])    
    j_sl = mano_out.joints + mano_param[prefix+'tsl']
    v_sl = mano_out.verts + mano_param[prefix+'tsl']

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v_sl[0].cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
    
    point_cloud = mesh.sample_points_uniformly(number_of_points=25000)
    points = np.asarray(point_cloud.points)
    
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_camera = (extrinsic @ points.T).T
    
    points_camera = points_camera[:, :3] / points_camera[:, 3:]

    v_2d_with_depth = (intrinsic @ points_camera.T).T
    v_2d = v_2d_with_depth[:, :2] / v_2d_with_depth[:, 2:]
    v_2d_depth = v_2d_with_depth[:, 2]
    
    # Transfer joints
    j = j_sl.clone().cpu().numpy()[0]
    
    j = np.concatenate([j, np.ones((j.shape[0], 1))], axis=1)
    j_camera = (extrinsic @ j.T).T
    j_camera = j_camera[:, :3] / j_camera[:, 3:]

    j_2d_with_depth = (intrinsic @ j_camera.T).T
    j_2d = j_2d_with_depth[:, :2] / j_2d_with_depth[:, 2:]
    j_2d_depth = j_2d_with_depth[:, 2]
    
    return v_2d, v_2d_depth, j_2d, j_2d_depth

def load_object_data(pcd, transf, extrinsic, intrinsic):
    points = np.asarray(pcd.points)
    points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    transformed_points_homo = (extrinsic @ transf @ points_homo.T).T
    transformed_points = transformed_points_homo[:, :3] / transformed_points_homo[:, 3:]
    
    two_d_points_with_depth = (intrinsic @ transformed_points.T).T
    two_d_points = two_d_points_with_depth[:, :2] / two_d_points_with_depth[:, 2:]
    two_d_points_depth = two_d_points_with_depth[:, 2]
    
    return two_d_points, two_d_points_depth

def load_object_data_mesh(mesh, transf, extrinsic, intrinsic):
    pcd = mesh.sample_points_uniformly(number_of_points=25000)
    return load_object_data(pcd, transf, extrinsic, intrinsic)
      
def draw_points_on_image(image, points):
    draw = ImageDraw.Draw(image)
    for point in points:
        draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill='red')
    
    # Save the image
    image.save('output.png')
    
def transfer_2d_points_to_mask(points, res, label=1, radius=2):
    """
    将2D散点投影为稠密mask：以小圆盘而不是单像素填充，减少空洞/黑点。
    """
    h, w = res
    mask = np.zeros((h, w), dtype=np.int32)
    if points is None or len(points) == 0:
        return mask
    for point in points:
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(mask, (x, y), int(radius), int(label), -1)
    return mask

def transfer_2d_points_to_depth(points, ori_depth, res):
    depth = np.ones(res) * 100000

    for i, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < res[1] and 0 <= y < res[0]:
            depth[y, x] = ori_depth[i]

    return depth


# def merge_hand_keypoints(left_hand_keypoints_image, right_hand_keypoints_image, left_deeper):
#     assert left_hand_keypoints_image.shape == right_hand_keypoints_image.shape, "Images must have the same shape"

#     height, width, channels = left_hand_keypoints_image.shape

#     result_image = np.zeros_like(left_hand_keypoints_image)
    
#     for y in range(height):
#         for x in range(width):
#             left_pixel = left_hand_keypoints_image[y, x]
#             right_pixel = right_hand_keypoints_image[y, x]
            
#             if np.all(left_pixel == [0, 0, 0]) and np.all(right_pixel == [0, 0, 0]):
#                 result_image[y, x] = [0, 0, 0]  
#             elif np.all(left_pixel == [0, 0, 0]):
#                 result_image[y, x] = right_pixel 
#             elif np.all(right_pixel == [0, 0, 0]):
#                 result_image[y, x] = left_pixel  
#             else:
#                 result_image[y, x] = left_pixel if not left_deeper[y, x] else right_pixel
                
#     return result_image

import numpy as np

def merge_hand_keypoints(left_hand_keypoints_image, right_hand_keypoints_image, left_deeper):
    # 确保输入的两个图像具有相同的形状
    assert left_hand_keypoints_image.shape == right_hand_keypoints_image.shape, "Images must have the same shape"

    # 获取图像的高度、宽度和通道数
    height, width, channels = left_hand_keypoints_image.shape

    # 创建一个空的结果图像
    result_image = np.zeros_like(left_hand_keypoints_image)

    # 创建布尔数组，表示哪些像素是零值
    left_is_zero = np.all(left_hand_keypoints_image == [0, 0, 0], axis=-1)
    right_is_zero = np.all(right_hand_keypoints_image == [0, 0, 0], axis=-1)

    # 合并条件
    result_image[left_is_zero & right_is_zero] = [0, 0, 0]  # 两者都是零
    result_image[~left_is_zero & right_is_zero] = left_hand_keypoints_image[~left_is_zero & right_is_zero]  # 左边不为零，右边为零
    result_image[left_is_zero & ~right_is_zero] = right_hand_keypoints_image[left_is_zero & ~right_is_zero]  # 左边为零，右边不为零
    
    # 对于其他情况，根据 `left_deeper` 选择左或右的像素
    
    if left_deeper[~left_is_zero & ~right_is_zero].any():
        result_image[~left_is_zero & ~right_is_zero] = np.where(left_deeper[~left_is_zero & ~right_is_zero][:, None],
                                                                right_hand_keypoints_image[~left_is_zero & ~right_is_zero],
                                                                left_hand_keypoints_image[~left_is_zero & ~right_is_zero])
    
    return result_image


def merge_object_hand_masks(object_masks, object_depths, left_hand_mask, 
                            right_hand_mask, left_hand_depth, right_hand_depth):
    """
    Hand Label : 1
    Object Label : 2-120
    """
    
    height, width = left_hand_mask.shape
    result_mask = np.zeros((height, width), dtype=np.uint8)
    mask_tensor = np.stack(object_masks + [left_hand_mask, right_hand_mask], axis=0)
    depth_tensor = np.stack(object_depths + [left_hand_depth, right_hand_depth], axis=0)
    
    # for y in range(height):
    #     for x in range(width):
    #         if np.all(mask_tensor[:, y, x] == 0):
    #             result_mask[y, x] = 0
    #         else:
    #             min_channel = np.argmin(depth_tensor[:, y, x])
    #             result_mask[y, x] = mask_tensor[min_channel, y, x]
    
    # 预计算 mask_tensor 是否为零的掩码
    non_zero_mask = np.any(mask_tensor != 0, axis=0)

    # 预先计算 depth_tensor 的最小值索引
    min_depth_indices = np.argmin(depth_tensor, axis=0)

    # 创建 result_mask 进行填充
    result_mask = np.zeros((height, width), dtype=np.uint8)

    # 对每个位置进行批量处理
    result_mask[non_zero_mask] = mask_tensor[min_depth_indices[non_zero_mask], non_zero_mask]

    return result_mask

def postprocess_label_mask(label_mask, min_area=50, k=3):
    """
    按类别进行形态学闭运算并移除小连通域，进一步消除黑点和噪声。
    """
    if label_mask is None:
        return label_mask
    kernel = np.ones((k, k), np.uint8)
    out = np.zeros_like(label_mask)
    unique_labels = np.unique(label_mask)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        binm = (label_mask == lbl).astype(np.uint8) * 255
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, kernel, iterations=1)
        num, cc, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
        keep = np.zeros_like(binm)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[cc == i] = 255
        out[keep > 0] = lbl
    return out

def get_conditions(seq_id):
    label = pickle.load(open(os.path.join('data/anno_preview', seq_id), 'rb'))
    # breakpoint()
    mocap_frame_id_list = label['mocap_frame_id_list']
    video_frame_id_list = label['frame_id_list']
    frame_id_list = list(set(mocap_frame_id_list) & set(video_frame_id_list))
    frame_id_list.sort()
    
    # Get Objecy
    object_meshes = {}
    object_id_list = label['obj_list']
    for object_id in object_id_list:
        object_file = glob.glob(f'data/object_repair/align_ds/{object_id}/*.obj')[0]
        object_mesh = o3d.io.read_triangle_mesh(object_file)
        object_meshes[object_id] = object_mesh
    
    hand_keypoints_list = []
    seg_mask_list = []
    colored_mask_list = []
    
    for frame_id in tqdm(frame_id_list):
        extrinsic = np.array(label['cam_extr']['allocentric_right'][frame_id])
        intrinsic = np.array(label['cam_intr']['allocentric_right'][frame_id])

        # Get Hand Mask
        mano_param = label['raw_mano'][frame_id]
        import time
        start = time.time()
        left_hand_2d, left_hand_depth, left_hand_joints_2d, left_hand_joints_depth = \
                        load_hand_data(mano_param, False, extrinsic, intrinsic)
                        
        print('load left hand data', time.time() - start)
        start = time.time()
        
        left_hand_mask = transfer_2d_points_to_mask(left_hand_2d, (480, 848))
        left_hand_depth_image = transfer_2d_points_to_depth(left_hand_2d, left_hand_depth, (480, 848))
        
        print('transfer left hand data', time.time() - start)

        right_hand_2d, right_hand_depth, right_hand_joints_2d, right_hand_joints_depth = \
                        load_hand_data(mano_param, True, extrinsic, intrinsic)   
        right_hand_mask = transfer_2d_points_to_mask(right_hand_2d, (480, 848))
        right_hand_depth_image = transfer_2d_points_to_depth(right_hand_2d, right_hand_depth, (480, 848))

        left_hand_keypoints_image = showHandJoints(np.zeros((480, 848, 3)), left_hand_joints_2d, left_hand_joints_depth, is_right=False)
        right_hand_keypoints_image = showHandJoints(np.zeros((480, 848, 3)), right_hand_joints_2d, right_hand_joints_depth, is_right=True)

        
        # Merge Hand Mask
        start = time.time()
        hand_keypoints_image = merge_hand_keypoints(left_hand_keypoints_image, 
                                                    right_hand_keypoints_image,
                                                    left_hand_depth_image > right_hand_depth_image)

        hand_keypoints_list.append(hand_keypoints_image)
        print('Merge Hand Keypoints', time.time() - start)
        
        # Get Object Mask
        start = time.time()
        object_masks = []
        object_depths = []
        count = 2
        for object_id, object_mesh in object_meshes.items():
            object_transf = label['obj_transf'][object_id][frame_id]
            object_2d, object_depth = load_object_data_mesh(object_mesh, object_transf, extrinsic, intrinsic)
            object_mask = transfer_2d_points_to_mask(object_2d, (480, 848), label=count)
            object_depth_image = transfer_2d_points_to_depth(object_2d, object_depth, (480, 848))
            object_masks.append(object_mask)
            object_depths.append(object_depth_image)
            count += 1
        print('Get Object Mask', time.time() - start)
        # Merge mask
        
        start = time.time()
        merged_mask = merge_object_hand_masks(
            object_masks, object_depths, left_hand_mask, right_hand_mask, left_hand_depth_image, right_hand_depth_image
        )
        merged_mask = postprocess_label_mask(merged_mask, min_area=50, k=3)
        seg_mask_list.append(merged_mask)
        colored_mask_list.append(convert_gray_to_color(merged_mask))
        print('Merge Mask', time.time() - start)
        # cv2.imwrite('output.png', convert_gray_to_color(merged_mask))
        # breakpoint()
    
    # Cat to video
    colored_mask_writer = cv2.VideoWriter(f'output_seg.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (848, 480))
    for seg_mask in colored_mask_list:
        colored_mask_writer.write(seg_mask.astype(np.uint8))
    colored_mask_writer.release()
    
    hand_keypoints_video_writer = cv2.VideoWriter(f'output_hand.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (848, 480))
    for hand_keypoints in hand_keypoints_list:
        hand_keypoints_video_writer.write(hand_keypoints.astype(np.uint8))
    hand_keypoints_video_writer.release()


def alpha_blend(video1, video2, output_path):
    # Alpha-blende 2 videos
    merged_video = []
    for frame1, frame2 in zip(video1, video2):
        alpha = 0.5
        blended_frame = cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)
        merged_video.append(blended_frame)

    merge_video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (848, 480))
    for frame in merged_video:
        merge_video_writer.write(frame.astype(np.uint8))
    merge_video_writer.release()
    
    pass

def mask_background(video, mask_video, output_path):
    video_reader = cv2.VideoCapture(video)
    mask_reader = cv2.VideoCapture(mask_video)

    rgb_frames, mask_frames = [], []
    while video_reader.isOpened() and mask_reader.isOpened():
        ret1, frame1 = video_reader.read()
        ret2, frame2 = mask_reader.read()

        if not ret1 or not ret2:
            break
        
        frame1 = cv2.resize(frame1, (848, 480))
        frame2 = cv2.resize(frame2, (848, 480))
        
        rgb_frames.append(frame1)
        mask_frames.append(frame2)

    video_reader.release()
    mask_reader.release()
    
    masked_frames = []
    for rgb_frame, mask_frame in zip(rgb_frames, mask_frames):
        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        mask_frame = cv2.threshold(mask_frame, 30, 255, cv2.THRESH_BINARY)[1]

        masked_frame = cv2.bitwise_and(rgb_frame, rgb_frame, mask=mask_frame)
        masked_frames.append(masked_frame)

    masked_frame_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (848, 480))

    for frame in masked_frames:
        if frame.shape[0] != 480 or frame.shape[1] != 848:
            breakpoint()
        frame = cv2.resize(frame, (848, 480))
        masked_frame_writer.write(frame)

    masked_frame_writer.release()
    

def clip_videos(video_path, output_root, video_name, frame_num):
    video_reader = cv2.VideoCapture(video_path)
    frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    
    num_clip_videos = frame_count // frame_num
    
    for i in range(num_clip_videos):
        # video_name = os.path.basename(video_path)
        os.makedirs(os.path.join(output_root, str(i)), exist_ok=True)
        output_path = os.path.join(output_root, str(i), video_name)
        
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (720, 480))
        
        for j in range(frame_num):
            ret, frame = video_reader.read()
            if not ret:
                break
            frame = cv2.resize(frame, (720, 480))
            video_writer.write(frame)

        video_writer.release()

def get_conditions_camera(seq_id, cam, start_frame_id, end_frame_id, frame_id_list, anno_root="/share/project/public_data/scenes/HOI-Mixed/OakInk-v2/anno_preview"):
    label = pickle.load(open(os.path.join(anno_root, f'{seq_id}.pkl'), 'rb'))
    cam_def = label['cam_def']

    frame_id_list = frame_id_list[start_frame_id:end_frame_id]

    # Get Objecy
    object_meshes = {}
    object_id_list = label['obj_list']
    for object_id in object_id_list:
        object_file = glob.glob(f'/share/project/public_data/scenes/HOI-Mixed/OakInk-v2/object_repair/align_ds/{object_id}/*.obj')[0]
        object_mesh = o3d.io.read_triangle_mesh(object_file)
        object_meshes[object_id] = object_mesh
    
    hand_keypoints_list = []
    seg_mask_list = []
    colored_mask_list = []
    
    for frame_id in tqdm(frame_id_list):
        extrinsic = np.array(label['cam_extr'][cam_def[cam]][frame_id])
        intrinsic = np.array(label['cam_intr'][cam_def[cam]][frame_id])

        # Get Hand Mask
        mano_param = label['raw_mano'][frame_id]
        # import time
        # start = time.time()
        left_hand_2d, left_hand_depth, left_hand_joints_2d, left_hand_joints_depth = \
                        load_hand_data(mano_param, False, extrinsic, intrinsic)
                        
        # print('load left hand data', time.time() - start)
        # start = time.time()
        
        left_hand_mask = transfer_2d_points_to_mask(left_hand_2d, (480, 848))
        left_hand_depth_image = transfer_2d_points_to_depth(left_hand_2d, left_hand_depth, (480, 848))
        
        # print('transfer left hand data', time.time() - start)

        right_hand_2d, right_hand_depth, right_hand_joints_2d, right_hand_joints_depth = \
                        load_hand_data(mano_param, True, extrinsic, intrinsic)   
        right_hand_mask = transfer_2d_points_to_mask(right_hand_2d, (480, 848))
        right_hand_depth_image = transfer_2d_points_to_depth(right_hand_2d, right_hand_depth, (480, 848))

        left_hand_keypoints_image = showHandJoints(np.zeros((480, 848, 3)), left_hand_joints_2d, left_hand_joints_depth, is_right=False)
        right_hand_keypoints_image = showHandJoints(np.zeros((480, 848, 3)), right_hand_joints_2d, right_hand_joints_depth, is_right=True)
        
        # Merge Hand Mask
        # start = time.time()
        hand_keypoints_image = merge_hand_keypoints(left_hand_keypoints_image, 
                                                    right_hand_keypoints_image,
                                                    left_hand_depth_image > right_hand_depth_image)

        hand_keypoints_list.append(hand_keypoints_image)
        # print('Merge Hand Keypoints', time.time() - start)
        
        # Get Object Mask
        # start = time.time()
        object_masks = []
        object_depths = []
        with open("utils/object_id.json", "r") as f_1:
            object_id_json = json.load(f_1)

        for object_id, object_mesh in object_meshes.items():
            object_transf = label['obj_transf'][object_id][frame_id]
            object_2d, object_depth = load_object_data_mesh(object_mesh, object_transf, extrinsic, intrinsic)
            object_mask = transfer_2d_points_to_mask(object_2d, (480, 848), label=object_id_json[object_id])
            object_depth_image = transfer_2d_points_to_depth(object_2d, object_depth, (480, 848))
            object_masks.append(object_mask)
            object_depths.append(object_depth_image)
        # print('Get Object Mask', time.time() - start)
        # Merge mask
        
        # start = time.time()
        merged_mask = merge_object_hand_masks(
            object_masks, object_depths, left_hand_mask, right_hand_mask, left_hand_depth_image, right_hand_depth_image
        )
        merged_mask = postprocess_label_mask(merged_mask, min_area=50, k=3)
        seg_mask_list.append(merged_mask)
        colored_mask_list.append(convert_gray_to_color(merged_mask))
        # print('Merge Mask', time.time() - start)
        # cv2.imwrite('output.png', convert_gray_to_color(merged_mask))
        # breakpoint()

    # output_dir = f'videos/{seq_id}/{cam}'
    # os.makedirs(output_dir, exist_ok=True)    
    # Cat to video
    # colored_mask_writer = cv2.VideoWriter(f'{output_dir}/output_seg.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (848, 480))
    # for seg_mask in colored_mask_list:
    #     colored_mask_writer.write(seg_mask.astype(np.uint8))
    # colored_mask_writer.release()
    
    # hand_keypoints_video_writer = cv2.VideoWriter(f'{output_dir}/output_hand.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (848, 480))
    # for hand_keypoints in hand_keypoints_list:
    #     hand_keypoints_video_writer.write(hand_keypoints.astype(np.uint8))
    # hand_keypoints_video_writer.release()

    return hand_keypoints_list, seg_mask_list, colored_mask_list
    
def get_seq_ids(filelist_path):
    seq_ids = []
    with open(filelist_path, 'r') as f:
        for line in f.readlines():
            if line.split('/')[-4] not in seq_ids:
                seq_ids.append(line.split('/')[-4])
    return seq_ids

def get_frame_index(data_dir):
    frame_index = []
    for frame_path in os.listdir(data_dir):
        frame_index.append(int(frame_path.split('.')[0]))
    frame_index.sort()
    return frame_index, len(frame_index)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--filelist_path", type=str, help="Path to the video filelist")
    parser.add_argument("--video_len", type=int, default=49, help="Length of the video chunk")
    parser.add_argument("--root_dir", type=str, help="Root directory of the dataset")
    args = parser.parse_args()

    hand_output_root_dir = os.path.join(args.root_dir, "processed_hand_keypoints")
    seg_output_root_dir = os.path.join(args.root_dir, "processed_seg_mask")
    seq_ids = get_seq_ids(args.filelist_path)

    for k, seq_id in enumerate(seq_ids):
        print(f"Processing {k} / {len(seq_ids)}: {seq_id}")
        seq_root_dir = os.path.join(args.root_dir, "data", seq_id)
        cameras = os.listdir(seq_root_dir)
        cameras.sort()

        for cam in cameras:
            if not os.path.exists(os.path.join(args.root_dir, "videos", seq_id, cam)):
                continue

            frame_index, frame_num = get_frame_index(os.path.join(args.root_dir, "data", seq_id, cam))
            video_chunk_num = frame_num // args.video_len

            for i in range(video_chunk_num):
                if not os.path.exists(os.path.join(args.root_dir, "videos", seq_id, cam)):
                    print(f"Video {seq_id} {cam} {i} not found")
                    continue

                hand_keypoints_list, seg_mask_list, colored_mask_list = get_conditions_camera(seq_id, cam, i * video_len, (i + 1) * video_len, frame_index)
                
                hand_output_dir = os.path.join(hand_output_root_dir, seq_id, cam, str(i))
                seg_output_dir = os.path.join(seg_output_root_dir, seq_id, cam, str(i))
                
                os.makedirs(hand_output_dir, exist_ok=True)
                os.makedirs(seg_output_dir, exist_ok=True)

                hand_video_writer = cv2.VideoWriter(f'{hand_output_dir}/output_hand.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (848, 480))
                for frame in hand_keypoints_list:
                    hand_video_writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
                hand_video_writer.release()

                seg_video_writer = cv2.VideoWriter(f'{seg_output_dir}/output_seg.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (848, 480))
                for frame in colored_mask_list:
                    seg_video_writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
                seg_video_writer.release()
