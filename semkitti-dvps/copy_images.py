import os
import shutil
import multiprocessing as mp


inp_dir = 'kitti/sequences'
out_dir = 'video_sequence'


def process(seq_id):
    split = 'val' if seq_id == 8 else 'train'
    seq_dir = os.path.join(inp_dir, '{:02d}'.format(seq_id), 'image_2')
    filenames = os.scandir(seq_dir)
    filenames = [name.name for name in filenames]
    for name in filenames:
        inp_path = os.path.join(seq_dir, name)
        new_name = '{:06d}_{}_leftImg8bit.png'.format(seq_id, name[:6])
        out_path = os.path.join(out_dir, split, new_name)
        shutil.copyfile(inp_path, out_path)


seq_ids = range(11)
with mp.Pool(processes=11) as p:
    p.map(process, seq_ids)
