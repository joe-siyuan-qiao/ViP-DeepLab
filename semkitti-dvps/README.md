## SemKITTI-DVPS

Please download the annotations from [link](https://drive.google.com/file/d/1cCn1yeu2dmT4CnvkOyItxARFBXpDdu-0/view?usp=sharing)
and extract it to here. It will be a folder named `video_sequence`.
Then, download the KITTI odometry color images from
[link](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
and place them in the structure as `kitti/sequences/00/image_2/000000.png`.
After downloading the images, run the following script to copy them into
`video_sequence`.
```bash
python copy_images.py
```
After the above procedures, SemKITTI-DVPS is ready and located in the folder
`video_sequence`.
The filenames of the depth annotations include the focal length in the last
segment. For example, the focal length of
`000000_000000_depth_718.8560180664062.png` is 718.8560180664062.
Depth maps are multiplied by 256 and saved as uint16 PNG images.

To evaluate the DVPQ performance,
```bash
python eval_dvpq.py --pan_dir PAN_DIR --depth_dir DEPTH_DIR --eval_frames
{1,5,10,20} --depth_thres {0.5,0.25,0.1}
```
The output will be DVPQ, DVPQ-Th, and DVPQ-St.
The script assumes that files in PAN_DIR with names `*cat.png` and `*ins.png`
store the classes and the instance IDs for the input images `*`, respectively.
The predictions and groundtruth are sorted according to their names for
matching.
