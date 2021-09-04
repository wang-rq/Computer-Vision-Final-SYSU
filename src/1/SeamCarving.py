import numpy as np
import cv2
from config import *

class SeamCarving:
    def __init__(self, src_file, dest_file, mid_imgs_path, dest_heigh, dest_wid, mask_file=''):
        # 读取图片
        self.in_image = cv2.imread(src_file).astype(np.float64)
        self.mask = cv2.imread(mask_file, 0).astype(np.float64)
        self.mid_imgs_path = mid_imgs_path

        self.in_heigh, self.in_wid = self.in_image.shape[: 2]
        self.dest_heigh, self.dest_wid = dest_heigh, dest_wid

        self.cur_img = np.copy(self.in_image)

        # 定义计算能量图的 kernel
        self.kernel_x = np.array(
            [[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array(
            [[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array(
            [[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)

        # 保护前景的常数
        self.constant = 1000

        # 进行 seam_carving
        self.seam_carving()  
        # 保存结果
        self.save_img(dest_file)  

    def seam_carving(self):
        # 计算需要删除的行列数
        delta_row = self.in_heigh - self.dest_heigh
        delta_col = self.in_wid - self.dest_wid

        # 纵向删除
        self.seams_removal(delta_col, 1)

        # 旋转图片，相当于横向删除
        self.cur_img = self.rotate_image(self.cur_img, 1)
        self.mask = self.rotate_mask(self.mask, 1)
        self.seams_removal(delta_row, 0)
        self.cur_img = self.rotate_image(self.cur_img, 0)

    def seams_removal(self, delta_col, iscol):
        for iter in range(delta_col):
            # 计算能量图
            energy_map = self.calculate_energy_map()
            # 遮罩保护区域
            energy_map[np.where(self.mask > 0)] *= self.constant
            # 计算累积能量图
            cumulative_map = self.dynamic_forward_cumu(energy_map)
            seam_idx = self.find_seam(cumulative_map)

            # 将 seam 设置为绿色便于展示结果
            self.show_seam(seam_idx)
            # 保存标记 seam 后的中间过程图片
            self.save_seam_mid(iscol, iter)
            # 删除 seam
            self.delete_seam(seam_idx)
            # 保存删除 seam 后的中间过程图片
            self.save_delete_mid(iscol, iter)
            # 将前景图片也同样缩小一列
            self.delete_seam_on_mask(seam_idx)

    # 保存标记 seam 后的中间过程图片
    def save_seam_mid(self, iscol, iter):
        if iscol == 1:
            self.save_img(self.mid_imgs_path + f'col-{iter}-1.png')
        else:
            temp = self.rotate_image(self.cur_img, 0)
            cv2.imwrite(self.mid_imgs_path + f'row-{iter}-1.png', temp.astype(np.uint8))
    
    # 保存删除 seam 后的中间过程图片
    def save_delete_mid(self, iscol, iter):
        if iscol == 1:
            self.save_img(mid_imgs_path + f'col-{iter}-2.png')
        else:
            temp = self.rotate_image(self.cur_img, 0)
            cv2.imwrite(mid_imgs_path + f'row-{iter}-2.png', temp.astype(np.uint8))

    # 计算能量图
    def calculate_energy_map(self):
        # 分开 b,g,r
        b, g, r = cv2.split(self.cur_img)
        # b,g,r 分别计算，使用 scharr 算子
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        # 三个计算结果相加
        return b_energy + g_energy + r_energy

    # 用动态规划法计算能量累加
    def dynamic_forward_cumu(self, energy_map):
        matrix_x = self.calculate_neighbor_matrix(self.kernel_x)
        matrix_y_left = self.calculate_neighbor_matrix(self.kernel_y_left)
        matrix_y_right = self.calculate_neighbor_matrix(self.kernel_y_right)

        m, n = energy_map.shape
        res = np.copy(energy_map)
        # 从第二行开始，动态规划计算最小值
        for row in range(1, m):
            for col in range(n):
                if col == 0:
                    e_right = res[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = res[row - 1, col] + matrix_x[row - 1, col]
                    res[row, col] = energy_map[row, col] + min(e_right, e_up)
                elif col == n - 1:
                    e_left = res[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_up = res[row - 1, col] + matrix_x[row - 1, col]
                    res[row, col] = energy_map[row, col] + min(e_left, e_up)
                else:
                    e_left = res[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_right = res[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = res[row - 1, col] + matrix_x[row - 1, col]
                    res[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
        return res

    # 计算邻居矩阵
    def calculate_neighbor_matrix(self, kernel):
        b, g, r = cv2.split(self.cur_img)
        res = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + np.absolute(cv2.filter2D(r, -1, kernel=kernel))
        return res

    # 从后往前寻找 seam 的点
    def find_seam(self, cumulative_map):
        m, n = cumulative_map.shape
        res = np.zeros((m,), dtype=np.uint32)
        res[-1] = np.argmin(cumulative_map[-1])
        for row in range(m - 2, -1, -1):
            prv_x = res[row + 1]
            if prv_x == 0:
                res[row] = np.argmin(cumulative_map[row, : 2])
            else:
                res[row] = np.argmin(
                    cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
        return res

    # 将剪裁线 seam 标出绿色
    def show_seam(self, seam_idx):
        m, _ = self.cur_img.shape[: 2]
        for row in range(m):
            col = seam_idx[row]
            self.cur_img[row, col, 0] = 0
            self.cur_img[row, col, 1] = 255
            self.cur_img[row, col, 2] = 0

    # 删除 seam
    def delete_seam(self, seam_idx):
        m, n = self.cur_img.shape[: 2]
        res = np.zeros((m, n - 1, 3))
        for row in range(m):
            col = seam_idx[row]
            for i in range(3):
                res[row, :, i] = np.delete(self.cur_img[row, :, i], [col])
        self.cur_img = np.copy(res)

    # 删除 mask 上的 seam
    def delete_seam_on_mask(self, seam_idx):
        m, n = self.mask.shape
        res = np.zeros((m, n - 1))
        for row in range(m):
            col = seam_idx[row]
            res[row, :] = np.delete(self.mask[row, :], [col])
        self.mask = np.copy(res)

    # 将图片旋转 90 度
    def rotate_image(self, image, ccw):
        m, n, ch = image.shape
        res = np.zeros((n, m, ch))
        if ccw:
            image_flip = np.fliplr(image)
            for c in range(ch):
                for row in range(m):
                    res[:, row, c] = image_flip[row, :, c]
        else:
            for c in range(ch):
                for row in range(m):
                    res[:, m - 1 - row, c] = image[row, :, c]
        return res

    # 将 mask 的图片旋转 90 度
    def rotate_mask(self, mask, ccw):
        m, n = mask.shape
        res = np.zeros((n, m))
        if ccw:
            image_flip = np.fliplr(mask)
            for row in range(m):
                res[:, row] = image_flip[row, :]
        else:
            for row in range(m):
                res[:, m - 1 - row] = mask[row, :]
        return res

    # 保存图片
    def save_img(self, filename):
        cv2.imwrite(filename, self.cur_img.astype(np.uint8))