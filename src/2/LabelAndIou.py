import numpy as np
import imageio
import os
import natsort
from segment import *
from config import *

# 给分割好的图像做标记
def label(k, sigma, min, src_filename, gt_filename):
    src_image = imageio.imread(src_filename)
    gt_image = imageio.imread(gt_filename)

    u, num_cc, colored = segment(src_image, sigma, k, min, min_num_sets, max_num_sets)

    heigh, wid, _ = src_image.shape
    comps = u.components()

    # 求每一块中是前景的像素数量
    fore_nums = [0] * num_cc
    for y in range(heigh):
        for x in range(wid):
            comp = u.find(y * wid + x)
            if gt_image[y,x,0]==255 and gt_image[y,x,1]==255 and gt_image[y,x,2]==255:
                fore_nums[comps.index(comp)] += 1

    # 求标记 list
    labels = [0] * num_cc
    for i in range(num_cc):
        if float(fore_nums[i]) / float(u.size(comps[i])) >= 0.5:
            labels[i] = 1 # 标记为前景
    
    # 求出标记后的分割图像
    labeled = np.zeros((heigh , wid, 3))
    for y in range(heigh):
        for x in range(wid):
            comp = u.find(y * wid + x)
            if labels[comps.index(comp)]:
                labeled[y, x, 0] = 255
                labeled[y, x, 1] = 255
                labeled[y, x, 2] = 255

    return colored, labeled

# 求 IOU
def IOU(gt_path, gt_seg_path, result_path):
    img_list = os.listdir(gt_path)
    img_list = natsort.natsorted(img_list)
    output = open(result_path+"result.txt", 'w')
    output.write(f"img \t\t IOU \n")
    print(f"img \t\t IOU")
    sum_iou = 0
    for i in img_list:
        _, type = os.path.splitext(i)
        if type == '.png':
            gt, gt_seg = imageio.imread(gt_path + i), imageio.imread(gt_seg_path + i)
            heigh, wid, _ = gt.shape
            intersection, union = 0.0, 0.0
            for y in range(heigh):
                for x in range(wid):
                    if gt[y,x,0]==255 and gt[y,x,1]==255 and gt[y,x,2]==255 and gt_seg[y,x,0]==255 and gt_seg[y,x,1]==255 and gt_seg[y,x,2]==255:
                        intersection += 1
                    if gt[y,x,0]==255 and gt[y,x,1]==255 and gt[y,x,2]==255 or gt_seg[y,x,0]==255 and gt_seg[y,x,1]==255 and gt_seg[y,x,2]==255:
                        union += 1
            iou = intersection/union
            sum_iou += iou
            output.write(f"{i}  \t {iou}\n")
            print(f"{i}  \t {iou}")
    avg_iou = sum_iou/len(img_list)
    output.write(f"average IOU: {avg_iou}\n")
    print(f"average IOU: {avg_iou}")
    output.close()
