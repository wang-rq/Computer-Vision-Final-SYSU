import os
import imageio
import natsort
from SeamCarving import *
from config import *

# 制作动图
def create_gif(path, gif_name, duration=0.1):   
    image_list = os.listdir(path)
    image_list = natsort.natsorted(image_list)
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(f'{path}/{image_name}'))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

if __name__ == '__main__':

    if not os.path.exists(src_file):
        print("Cannot find file", src_file)
        os._exit(1)
    if not os.path.exists(filename_mask):
        print("Cannot find file", filename_mask)
        os._exit(1)
    if not os.path.exists(mid_imgs_path):
        print("mkdir", mid_imgs_path)
        os.makedirs(mid_imgs_path)


    new_height = 50
    new_width = 80

    # seam carving
    print('Seam carving...')
    SeamCarving(src_file, dest_file, mid_imgs_path, new_height, new_width, mask_file=filename_mask)

    # 制作动图
    print('Creating gif...')
    create_gif(mid_imgs_path, gif_file, 0.1)