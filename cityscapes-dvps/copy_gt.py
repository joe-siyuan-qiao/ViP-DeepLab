import os
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()
split = args.split

src_dir = os.path.join('vps/data/cityscapes_vps', split, 'panoptic_inst')
dst_dir = os.path.join('gtFine', split)


for root, dirs, files in os.walk(src_dir):
    for name in files:
        place = name.split('_')[2]
        dst_name = name[10:]
        dst_name = dst_name.replace('final_mask', 'gtFine_instanceTrainIds')
        dst_name = dst_name.replace('gtFine_color', 'gtFine_instanceTrainIds')
        src_file = os.path.join(root, name)
        dst_file = os.path.join(dst_dir, place)
        if not os.path.isdir(dst_file):
            os.makedirs(dst_file)
        dst_file = os.path.join(dst_file, dst_name)
        shutil.copyfile(src_file, dst_file)
