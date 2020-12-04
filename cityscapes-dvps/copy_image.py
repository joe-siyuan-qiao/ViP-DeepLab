import shutil
import os
import multiprocessing as mp
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()
split = args.split

file_names = os.scandir(os.path.join('vps/data/cityscapes_vps', split, 'cls'))
file_names = [file_name.name for file_name in file_names]


def copy_image(file_name):
    file_name = os.path.basename(file_name)
    file_name = file_name[10:]
    file_name = file_name.replace('final_mask', 'leftImg8bit')
    file_name = file_name.replace('gtFine_color', 'leftImg8bit')
    place = file_name.split('_')[0]
    src_dir = os.path.join(
        'vps/data/leftImg8bit_sequence/val',
        place,
        file_name)
    dst_dir = os.path.join('leftImg8bit', split, place)
    os.makedirs(dst_dir, exist_ok=True)
    dst_dir = os.path.join(dst_dir, file_name)
    shutil.copyfile(src_dir, dst_dir)


N = mp.cpu_count()
with mp.Pool(processes=N) as p:
    p.map(copy_image, file_names)
