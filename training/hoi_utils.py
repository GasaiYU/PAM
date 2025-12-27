import numpy as np
import cv2
import sys
import math

import os

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
      [0, 0, 0]
  ]


def showHandJoints(imgInOrg, gtIn, gtIn3D=None, filename=None):
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
    if gtIn3D is not None:
        if gtIn3D[0, 2] < gtIn3D[-1, 2]: 
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
                                                (int(length / 2), 6),
                                                int(deg),
                                                0, 360, 1)
                    else:
                        polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                                (int(length / 2), 3),
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


if __name__== '__main__':
    frames_base = '/data115/video-diff/workspace/HOI-DiffusionAsShader/data/dexycb/20200709-subject-01/20200709_141754/836212060125'
    hand_pose_files = []
    for file in os.listdir(frames_base):
        if file.startswith('label'):
            hand_pose_files.append(file)
    
    kps = []
    hand_pose_files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    for file in hand_pose_files:
        label = np.load(os.path.join(frames_base, file))
        joint_2d = label['joint_2d'][0]
        if (joint_2d == -1).any():
            kps.append(np.zeros((480, 640, 3), dtype=np.uint8))
        else:
            kps.append(showHandJoints(np.zeros((480, 640, 3), dtype=np.uint8), joint_2d))

    out = cv2.VideoWriter('hand_pose.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for frame in kps:
        out.write(frame)
    out.release()

def convert_gray_to_color(gray_image, color_map=CLASS_PROTOCAL):
    gray_image[gray_image==255] = 149
    gray_image[gray_image==0] = 150
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

def mask_conditions(input_videos, mask_videos, black_threshold=-0.85):
    """
    Mask out regions in input_videos where mask_videos is (approximately) black across channels.

    Supports 4D tensors shaped (T, C, H, W) and 5D tensors shaped (B, T, C, H, W).
    """
    # mask_videos = (mask_videos + 1.0) * 127.5
    # mask_videos = mask_videos.cpu().numpy().astype(np.uint8)
    # mask_videos = mask_videos.transpose(0, 2, 3, 1)
    # cv2.imwrite("temp/mask_videos.png", mask_videos[0])
    # breakpoint()
    if input_videos.ndim not in (4, 5) or mask_videos.ndim not in (4, 5):
        raise ValueError("Expected 4D (T,C,H,W) or 5D (B,T,C,H,W) tensors for videos")

    # Determine channel dimension for both inputs
    input_channel_dim = 1 if input_videos.ndim == 4 else 2
    mask_channel_dim = 1 if mask_videos.ndim == 4 else 2

    # Validate temporal and spatial dimensions match (ignore channel counts)
    def dims_except(tensor, channel_dim):
        return tuple(d for i, d in enumerate(tensor.shape) if i != channel_dim)

    if dims_except(input_videos, input_channel_dim) != dims_except(mask_videos, mask_channel_dim):
        raise ValueError("Input and mask must match on all dims except channels")

    # Identify black pixels in the mask across channels with a tolerance
    # keepdim=True yields a (.., 1, H, W) mask we can expand to input channels
    is_black = (mask_videos <= black_threshold).all(dim=mask_channel_dim, keepdim=True)

    # Broadcast to input's channel count
    if is_black.shape[input_channel_dim] != 1:
        # Realign keepdim axis to match input layout when ndims differ
        # This happens only if input and mask have different ndims (should not in our use case),
        # but guard anyway by reshaping to singleton channel at the expected axis.
        # Build a view shape of ones and copy non-channel dims from input
        view_shape = [1] * input_videos.ndim
        for i in range(input_videos.ndim):
            if i == input_channel_dim:
                continue
            view_shape[i] = input_videos.shape[i]
        is_black = is_black.reshape(view_shape)

    mask = is_black.expand_as(input_videos)

    # Apply the mask
    masked_videos = input_videos.clone()
    masked_videos[mask] = -1.0
    return masked_videos
