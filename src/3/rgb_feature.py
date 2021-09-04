import os
import cv2
import numpy as np
from math import floor
from segment import *
from HashTable import *
from sklearn.decomposition import PCA
from config import *

# 计算颜色对比度特征
def get_rgb_hist_feature(img, mask=None):
    hist_b = cv2.calcHist([img], [0], mask, [512], [0, 256])
    hist_g = cv2.calcHist([img], [1], mask, [512], [0, 256])
    hist_r = cv2.calcHist([img], [2], mask, [512], [0, 256])
    hist = hist_b + hist_g + hist_r
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return list(np.array(hist).ravel())

# 生成遮罩 mask
def gen_mask(comp, djs, ht):
    mask = np.zeros(shape=(ht.h, ht.w), dtype=np.uint8)
    vertices = djs.all_vertices_in_comp(comp)
    for v in vertices:
        pix = ht.vertice2pix(v)
        mask[pix[0], pix[1]] = 255
        mask = mask.astype(np.uint8)
    return mask

# 将两个feature连接起来成为一个1024维的特征
def get_features(img, djs, ht):
    all_rgb_feature = get_rgb_hist_feature(img)
    features = []
    for comp in djs.components():
        mask = gen_mask(comp, djs, ht)
        comp_rgb_feature = get_rgb_hist_feature(img, mask)
        fvec = np.concatenate((comp_rgb_feature, all_rgb_feature))
        features.append(fvec)
    features = np.array(features)
    return features

# 生成标签
def get_label(ht, comps, gt_seg):
    y_train = []
    for comp in comps:
        (y, x) = ht.vertice2pix(comp)
        if gt_seg[y, x] == 255:
            y_train.append(1)
        else:
            y_train.append(0)
    return y_train

# 生成数据集
def gen_rgb_data_set(im_path, gt_seg_path):
    pic_list = os.listdir(im_path)
    data, label = [], []

    for (i, pic) in enumerate(pic_list):
        print(i + 1, "/", len(pic_list))
        img, gt_seg = cv2.imread(im_path+pic), cv2.imread(gt_seg_path+pic)
        gt_seg = cv2.cvtColor(gt_seg, cv2.COLOR_BGR2GRAY)

        djs, _, _ = segment(img, sigma, k, min, min_num_sets, max_num_sets)
        ht = vp_hash_table(img.shape[0], img.shape[1])

        features = get_features(img, djs, ht)
        for fvec in features:
            data.append(fvec)

        label = label + get_label(ht, djs.components(), gt_seg)
    pca = PCA(n_components=20)
    data = pca.fit_transform(data)
    data, label = np.array(data), np.array(label)
    return data, label


def generate_rgb_feature():
    print("training data making...")
    x_train, y_train = gen_rgb_data_set(im_path, gt_seg_path)
    print("train data done")
    print("test data making...")
    x_test, y_test = gen_rgb_data_set(test_im_path, test_gt_seg_path)
    print("test data done")
    np.save(train_file_path+"x_train_rgb.npy", x_train)
    np.save(train_file_path+"y_train.npy", y_train)
    np.save(test_file_path+"x_test_rgb.npy", x_test)
    np.save(test_file_path+"y_test.npy", y_test)
