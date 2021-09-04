from filter import *
from segment_graph import *
import time


# -------------------------------------------------------------------------------
# Segment an image:
# Returns a color image representing the segmentation.
#
# Inputs:
#           in_image: image to segment.
#           sigma: to smooth the image.
#           k: constant for threshold function.
#           min_size: minimum component size (enforced by post-processing stage).
#
# Returns:
#           num_ccs: number of connected components in the segmentation.
# -------------------------------------------------------------------------------


def segment(in_image, sigma, k, min_size, min_num_sets, max_num_sets):

    start_time = time.time()
    heigh, wid, band = in_image.shape

    smooth_red_band = smooth(in_image[:, :, 0], sigma)
    smooth_green_band = smooth(in_image[:, :, 1], sigma)
    smooth_blue_band = smooth(in_image[:, :, 2], sigma)

    # build graph
    edges_size = wid * heigh * 4
    edges = np.zeros(shape=(edges_size, 3), dtype=object)
    num = 0
    for y in range(heigh):
        for x in range(wid):
            if x < wid - 1:
                edges[num, 0] = int(y * wid + x)
                edges[num, 1] = int(y * wid + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y)
                num += 1
            if y < heigh - 1:
                edges[num, 0] = int(y * wid + x)
                edges[num, 1] = int((y + 1) * wid + x)
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x, y + 1)
                num += 1

            if (x < wid - 1) and (y < heigh - 2):
                edges[num, 0] = int(y * wid + x)
                edges[num, 1] = int((y + 1) * wid + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y + 1)
                num += 1

            if (x < wid - 1) and (y > 0):
                edges[num, 0] = int(y * wid + x)
                edges[num, 1] = int((y - 1) * wid + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y - 1)
                num += 1
    # Segment
    u = segment_graph(wid * heigh, num, edges, k)

    # post process small components
    for i in range(num):
        a = u.find(edges[i, 0])
        b = u.find(edges[i, 1])
        # 使得每一块最小 min_size 个像素，并限制分割块数大于 min_num_sets
        if (a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)) and u.num_sets() > min_num_sets:
            u.join(a, b)

    loose = 0
    stride = 5 # 设置宽松区间步长
    # 限制分割块数小于 max_num_sets，如果大于 min_num_sets，则合并较小的区域
    while(u.num_sets() > max_num_sets):
        loose += stride
        for i in range(num):
            a = u.find(edges[i, 0])
            b = u.find(edges[i, 1])
            if (a != b) and ((u.size(a) < min_size + loose) or (u.size(b) < min_size + loose)) and u.num_sets() >= min_num_sets:
                u.join(a, b)

    num_cc = u.num_sets()
    # print(num_cc)
    
    output = np.zeros(shape=(heigh, wid, 3))

    # pick random colors for each component
    colors = np.zeros(shape=(heigh * wid, 3))
    for i in range(heigh * wid):
        colors[i, :] = random_rgb()

    for y in range(heigh):
        for x in range(wid):
            comp = u.find(y * wid + x)
            output[y, x, :] = colors[comp, :]

    elapsed_time = time.time() - start_time
    print(
        "Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
            int(elapsed_time % 60)) + " seconds")
    
    return u, num_cc, output


