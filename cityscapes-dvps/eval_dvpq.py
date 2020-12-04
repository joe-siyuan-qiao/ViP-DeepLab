import numpy as np
from PIL import Image
import six
import os
import multiprocessing as mp
import pdb
import sys
import argparse


parser = argparse.ArgumentParser('')
parser.add_argument('--pan_dir', type=str, default='')
parser.add_argument('--depth_dir', type=str, default='')
parser.add_argument('--eval_frames', type=int, default=1)
parser.add_argument('--depth_thres', type=float, default=0)
args = parser.parse_args()


eval_frames = args.eval_frames
pred_dir = args.pan_dir
depth_dir = args.depth_dir
gt_dir = 'video_sequence/val'
depth_thres = args.depth_thres


def vpq_eval(element):
    preds, gts = element
    max_ins = 1000
    ign_id = 32
    offset = 256 * 256
    num_cat = 20

    assert isinstance(preds, list)
    assert isinstance(gts, list)
    assert len(preds) == len(gts)

    iou_per_class = np.zeros(num_cat, dtype=np.float64)
    tp_per_class = np.zeros(num_cat, dtype=np.float64)
    fn_per_class = np.zeros(num_cat, dtype=np.float64)
    fp_per_class = np.zeros(num_cat, dtype=np.float64)

    pred_ids = np.concatenate(preds, axis=1)
    gt_ids = np.concatenate(gts, axis=1)

    def _ids_to_counts(id_array):
        ids, counts = np.unique(id_array, return_counts=True)
        return dict(six.moves.zip(ids, counts))

    pred_areas = _ids_to_counts(pred_ids)
    gt_areas = _ids_to_counts(gt_ids)

    void_id = ign_id * max_ins
    ign_ids = {
        gt_id for gt_id in six.iterkeys(gt_areas)
        if (gt_id // max_ins) == ign_id
    }

    int_ids = gt_ids.astype(np.uint32) * offset + pred_ids.astype(np.uint32)
    int_areas = _ids_to_counts(int_ids)

    def prediction_void_overlap(pred_id):
        void_int_id = void_id * offset + pred_id
        return int_areas.get(void_int_id, 0)

    def prediction_ignored_overlap(pred_id):
        total_ignored_overlap = 0
        for _ign_id in ign_ids:
            int_id = _ign_id * offset + pred_id
            total_ignored_overlap += int_areas.get(int_id, 0)
        return total_ignored_overlap

    gt_matched = set()
    pred_matched = set()

    for int_id, int_area in six.iteritems(int_areas):
        gt_id = int_id // offset
        gt_cat = gt_id // max_ins
        pred_id = int_id % offset
        pred_cat = pred_id // max_ins
        if gt_cat != pred_cat:
            continue
        union = (
            gt_areas[gt_id] + pred_areas[pred_id] - int_area -
            prediction_void_overlap(pred_id)
        )
        iou = int_area / union
        if iou > 0.5:
            tp_per_class[gt_cat] += 1
            iou_per_class[gt_cat] += iou
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)

    for gt_id in six.iterkeys(gt_areas):
        if gt_id in gt_matched:
            continue
        cat_id = gt_id // max_ins
        if cat_id == ign_id:
            continue
        fn_per_class[cat_id] += 1

    for pred_id in six.iterkeys(pred_areas):
        if pred_id in pred_matched:
            continue
        if (prediction_ignored_overlap(pred_id) / pred_areas[pred_id]) > 0.5:
            continue
        cat = pred_id // max_ins
        fp_per_class[cat] += 1

    return (iou_per_class, tp_per_class, fn_per_class, fp_per_class)


def eval(element):
    max_ins = 1000
    num_frames = 6

    preds, gts, depth_preds, depth_gts = element
    preds = [np.array(Image.open(os.path.join(pred_dir, pred)))
             for pred in preds]
    cats = [pred[:, :, 0] for pred in preds]
    inds = [pred[:, :, 1] for pred in preds]
    preds = [cat.astype(np.uint32) * max_ins + ind.astype(np.uint32)
             for cat, ind in zip(cats, inds)]
    gts = [np.array(Image.open(os.path.join(gt_dir, name))) for name in gts]

    abs_rel = 0
    if depth_thres > 0:
        depth_preds = [np.array(Image.open(os.path.join(depth_dir, name)))
                       for name in depth_preds]
        depth_gts = [np.array(Image.open(os.path.join(gt_dir, name)))
                     for name in depth_gts]
        depth_pred_cat = np.concatenate(depth_preds, axis=1)
        depth_gt_cat = np.concatenate(depth_gts, axis=1)
        depth_mask = depth_gt_cat > 0
        abs_rel = np.mean(
            np.abs(
                depth_pred_cat[depth_mask] -
                depth_gt_cat[depth_mask]) /
            depth_gt_cat[depth_mask])
        for depth_pred, depth_gt, pred in zip(depth_preds, depth_gts, preds):
            depth_mask = depth_gt > 0
            pred_in_depth_mask = pred[depth_mask]
            ignored_pred_mask = (
                np.abs(
                    depth_pred[depth_mask] -
                    depth_gt[depth_mask]) /
                depth_gt[depth_mask]) > depth_thres
            pred_in_depth_mask[ignored_pred_mask] = 19 * max_ins
            pred[depth_mask] = pred_in_depth_mask

    def _gt_process(img, max_ins=1000):
        cat = img // 1000
        ins = img % 1000
        ids = cat * max_ins + ins
        return ids.astype(np.uint32)

    gts = [_gt_process(gt, max_ins) for gt in gts]
    results = []
    for i in range(num_frames - eval_frames + 1):
        pred_t = preds[i: i + eval_frames]
        gt_t = gts[i: i + eval_frames]
        result = vpq_eval([pred_t, gt_t])
        results.append(result)

    iou_per_class = np.stack([result[0] for result in results]).sum(axis=0)
    tp_per_class = np.stack([result[1] for result in results]).sum(axis=0)
    fn_per_class = np.stack([result[2] for result in results]).sum(axis=0)
    fp_per_class = np.stack([result[3] for result in results]).sum(axis=0)

    return (iou_per_class, tp_per_class, fn_per_class, fp_per_class, abs_rel)


def load_sequence(inp_lst):
    out_dict = dict()
    for inp in inp_lst:
        seq_id = inp.split('_')[0]
        if seq_id not in out_dict:
            out_dict[seq_id] = []
        out_dict[seq_id].append(inp)
    for seq_id in out_dict:
        out_dict[seq_id] = sorted(out_dict[seq_id])
    return out_dict


def main():
    # Load vps prediction
    pred_names = os.scandir(pred_dir)
    pred_names = [name.name for name in pred_names]
    preds = load_sequence(pred_names)

    # Load vps groundtruth
    gt_names = os.scandir(gt_dir)
    gt_names = [name.name for name in gt_names if 'gtFine' in name.name]
    gts = load_sequence(gt_names)

    # Load depth groundtruth
    depth_gt_names = os.scandir(gt_dir)
    depth_gt_names = [
        name.name for name in depth_gt_names if 'depth' in name.name]
    depth_gts = load_sequence(depth_gt_names)

    # Load depth prediction
    if depth_thres > 0:
        depth_pred_names = os.scandir(depth_dir)
        depth_pred_names = [name.name for name in depth_pred_names]
        depth_preds = load_sequence(depth_pred_names)

    # Evaluation
    all_files = []
    for seq_id in gts:
        depth_preds_seq_id = depth_preds[seq_id] if args.depth_thres > 0 else None
        all_files.append([preds[seq_id], gts[seq_id],
                          depth_preds_seq_id, depth_gts[seq_id]])
    N = mp.cpu_count() * 2
    with mp.Pool(processes=N) as p:
        results = p.map(eval, all_files)

    iou_per_class = np.stack([result[0] for result in results]).sum(axis=0)
    tp_per_class = np.stack([result[1] for result in results]).sum(axis=0)
    fn_per_class = np.stack([result[2] for result in results]).sum(axis=0)
    fp_per_class = np.stack([result[3] for result in results]).sum(axis=0)
    abs_rel = np.stack([result[4] for result in results]).mean(axis=0)
    epsilon = 0
    iou_per_class = iou_per_class[:19]
    tp_per_class = tp_per_class[:19]
    fn_per_class = fn_per_class[:19]
    fp_per_class = fp_per_class[:19]
    sq = iou_per_class / (tp_per_class + epsilon)
    rq = tp_per_class / (tp_per_class + 0.5 *
                         fn_per_class + 0.5 * fp_per_class + epsilon)
    pq = sq * rq
    spq = pq[:11]
    tpq = pq[11:]
    print(
        '{:.2f} {:.2f} {:.2f}'.format(
            pq.mean() * 100,
            tpq.mean() * 100,
            spq.mean() * 100))


if __name__ == '__main__':
    main()
