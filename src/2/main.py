import numpy as np
import imageio
import os
import natsort
from segment import *
from config import *
from LabelAndIou import *


if __name__ == "__main__":

    if not os.path.exists(src_path):
        print("Cannot find path", src_path)
        os._exit(1)
    if not os.path.exists(gt_path):
        print("Cannot find path", gt_path)
        os._exit(1)
    if not os.path.exists(result_path_gt_seg):
        print("mkdir", result_path_gt_seg)
        os.makedirs(result_path_gt_seg)
    if not os.path.exists(result_path_colored):
        print("mkdir", result_path_colored)
        os.makedirs(result_path_colored)
    if not os.path.exists(result_path):
        print("mkdir", result_path)
        os.makedirs(result_path)

    img_list = os.listdir(src_path)
    img_list = natsort.natsorted(img_list)
    for i in img_list:
        print("processing", i)
        colored, seg_image = label(k, sigma, min, src_path + i, gt_path + i)
        imageio.imwrite(result_path_colored + i, colored.astype(np.uint8))
        imageio.imwrite(result_path_gt_seg + i, seg_image.astype(np.uint8))
    
    IOU(gt_path, result_path_gt_seg, result_path)