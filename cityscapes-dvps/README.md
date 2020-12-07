## Cityscapes-DVPS

Please download the depth annotations from [link](https://drive.google.com/file/d/147esC0jEhWQCCEMOHj5nYGCPnuwmQrl4/view?usp=sharing)
and extract it to here. It will be a folder named `video_sequence`. Then,
```bash
git clone https://github.com/joe-siyuan-qiao/vps.git
```
and follow [DATASET.md](https://github.com/joe-siyuan-qiao/vps/blob/master/docs/DATASET.md)
to prepare Cityscapes-VPS.
Then, run the following scripts to generate __train__ and __val__ splits of
Cityscapes-DVPS:
```bash
python copy_image.py --split train
python copy_image.py --split val
python copy_gt.py --split train
python copy_gt.py --split val
python copy_video_sequence.py --split train
python copy_video_sequence.py --split val
```
After the above procedures, Cityscapes-DVPS is ready and located in the folder
`video_sequence`.
The panoptic maps stores `class * 1000 + instance_id`.
Depth maps are multiplied by 256 and saved as uint16 PNG images.

To evaluate the DVPQ performance,
```bash
python eval_dvpq.py --pan_dir PAN_DIR --depth_dir DEPTH_DIR --eval_frames
{1,2,3,4} --depth_thres {0.5,0.25,0.1}
```
The output will be DVPQ, DVPQ-Th, and DVPQ-St.
The predictions in PAN_DIR and DPETH_DIR are expected to have the same names as their inputs.
Panoptic predictions are assumed to be PNG images in 2-channel format where `[:,:,0]` stores the class and `[:,:,1]` stores the instance IDs.
