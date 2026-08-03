from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2
import torch
import torch.nn.functional as F
from core.inference import get_max_preds


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size * heatmap_height,
                           (num_joints + 1) * heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255) \
            .clamp(0, 255) \
            .byte() \
            .permute(1, 2, 0) \
            .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255) \
            .clamp(0, 255) \
            .byte() \
            .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


joints = torch.Tensor([[[1, 1, 1], [20, 20, 1], [30, 30, 1], [40, 40, 1]]])
heatmap_shape = (128, 128)
target_type = 'gaussian'
sigma = 3
num_boxes, num_joints, _ = joints.shape
target_weight = (joints[..., 2] > 0).float()

device = joints.device

map_w, map_h = heatmap_shape

assert target_type == 'gaussian', 'Only support gaussian map now!'

target = torch.zeros(num_boxes, num_joints, map_h, map_w).float().to(device)

# normalized [x, y] * [w, h]
joints_loc = torch.round(joints[..., :2] * joints.new([map_w, map_h])).int()
mu_x = joints_loc[..., 0]
mu_y = joints_loc[..., 1]

if target_type == 'gaussian':
    tmp_size = sigma * 3

    left = mu_x - tmp_size  # size: [num_box, num_kpts]
    right = mu_x + tmp_size + 1
    up = mu_y - tmp_size
    down = mu_y + tmp_size + 1

    # heatmap range
    img_x_min = torch.clamp(left, min=0)
    img_x_max = torch.clamp(right, max=map_w)
    img_y_min = torch.clamp(up, min=0)
    img_y_max = torch.clamp(down, max=map_h)

    # usable gaussian range
    gx_min = torch.clamp(-left, min=0)
    gx_max = img_x_max - left
    gy_min = torch.clamp(-up, min=0)
    gy_max = img_y_max - up

    is_out_bound = (left >= map_w) | (
            up >= map_h) | (right < 0) | (down < 0)

    # # Generate gaussian
    size = 2 * tmp_size + 1
    x = torch.arange(0, size, 1).float().to(device)
    y = x[:, None]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    for i in range(num_boxes):
        for j in range(num_joints):
            if is_out_bound[i, j]:
                continue
            if target_weight[i, j] > 0:
                target[i, j, img_y_min[i, j]: img_y_max[i, j],
                img_x_min[i, j]: img_x_max[i, j]] = g[gy_min[i, j]: gy_max[i, j], gx_min[i, j]: gx_max[i, j]]
print(target.shape)
input = torch.ones(1, 3, 128, 128)
save_batch_heatmaps(input, target, 'hm_gt.jpg')
