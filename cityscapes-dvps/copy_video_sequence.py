import os
import argparse
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='val')
args = parser.parse_args()
split = args.split


src_dir = os.path.join('video_sequence', split)
depth_names = os.scandir(src_dir)
depth_names = [name.name for name in depth_names]
depth_names = [name for name in depth_names if 'depth' in name]

for depth_name in depth_names:
    city = depth_name.split('_')[2]
    dst_img_name = depth_name.replace('depth', 'leftImg8bit')
    src_img_name = dst_img_name[14:]
    dst_img_path = os.path.join(src_dir, dst_img_name)
    src_img_path = os.path.join('leftImg8bit', split, city, src_img_name)
    shutil.copyfile(src_img_path, dst_img_path)
    dst_pan_name = depth_name.replace('depth', 'gtFine_instanceTrainIds')
    src_pan_name = dst_pan_name[14:]
    dst_pan_path = os.path.join(src_dir, dst_pan_name)
    src_pan_path = os.path.join('gtFine', split, city, src_pan_name)
    shutil.copyfile(src_pan_path, dst_pan_path)
