import numpy as np
from config import *
from sklearn.cluster import MiniBatchKMeans

# 标准化
def normalization(datingDatamat):
   max_arr = datingDatamat.max(axis=0)
   min_arr = datingDatamat.min(axis=0)
   ranges = max_arr - min_arr
   norDataSet = np.zeros(datingDatamat.shape)
   m = datingDatamat.shape[0]
   norDataSet = datingDatamat - np.tile(min_arr, (m, 1))
   norDataSet = norDataSet/np.tile(ranges,(m,1))
   return norDataSet

# 获得 word 的特征，即聚簇的平均特征
def get_vocabulary(file):
    rgb_feature_vectors = np.load(file)
    rgb_feature_vectors = normalization(rgb_feature_vectors)
    kmeans = MiniBatchKMeans(n_clusters = vocab_size, max_iter=500).fit(rgb_feature_vectors) # change max_iter for lower compute time
    vocabulary = np.vstack(kmeans.cluster_centers_)
    return vocabulary

# 计算点积相似度，并和原来的20维度特征做点积相似度，获得题目要求的全部70维特征
def get_full_feature(vocabulary, src_file, dest_file):
    rgb_feature_vectors = np.load(src_file)
    rgb_feature_vectors = normalization(rgb_feature_vectors)
    similarity = np.dot(rgb_feature_vectors, np.transpose(vocabulary))
    full_feature = np.concatenate((rgb_feature_vectors, normalization(similarity)), axis=1)
    np.save(dest_file, full_feature)
    return full_feature

# 生成完整的特征数据集
def generate_full_feature():
    rgb_train_file = train_file_path+"x_train_rgb.npy"
    rgb_test_file = test_file_path+"x_test_rgb.npy"
    full_train_file = train_file_path+"x_train_full.npy"
    full_test_file = test_file_path+"x_test_full.npy"

    vocabulary = get_vocabulary(rgb_train_file)

    get_full_feature(vocabulary, rgb_train_file, full_train_file)
    get_full_feature(vocabulary, rgb_test_file, full_test_file)

