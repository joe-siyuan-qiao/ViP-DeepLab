import numpy as np
from PIL import Image
import six
import os
import pdb
import sys
import time
import argparse
import yaml

import collections
from typing import MutableMapping, Sequence, Dict, Text, Any
import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser(description='')
parser.add_argument('--pan_dir', type=str, default='')
parser.add_argument('--depth_dir', type=str, default='')
parser.add_argument('--gt_dir', type=str, default='')
parser.add_argument('--out_dir', type=str, default='')
args = parser.parse_args()

pred_dir = args.pan_dir
depth_dir = args.depth_dir
gt_dir = args.gt_dir
out_dir = args.out_dir


def _update_dict_stats(stat_dict: MutableMapping[int, tf.Tensor],
                       id_array: tf.Tensor):
  """Updates a given dict with corresponding counts."""
  ids, _, counts = tf.unique_with_counts(id_array)
  for idx, count in zip(ids.numpy(), counts):
    if idx in stat_dict:
      stat_dict[idx] += count
    else:
      stat_dict[idx] = count


class DSTQuality(object):
  """Metric class for the Depth-aware Segmentation and Tracking Quality (DSTQ).

  The metric computes the geometric mean of three terms.
  - Association Quality: This term measures the quality of the track ID
      assignment for `thing` classes. It is formulated as a weighted IoU
      measure.
  - Segmentation Quality: This term measures the semantic segmentation quality.
      The standard class IoU measure is used for this.
  - Depth Inlier: This term measures the ratio of the depth inliers.

  Example usage:

  dstq_obj = DSTQuality(num_classes, things_list,
    ignore_label, max_instances_per_category, offset)
  dstq_obj.update_state(y_true_1, y_pred_1, depth_inlier)
  dstq_obj.update_state(y_true_2, y_pred_2, depth_inlier)
  ...
  result = dstq_obj.result().numpy()
  """

  def __init__(self,
               num_classes: int,
               things_list: Sequence[int],
               ignore_label: int,
               max_instances_per_category: int,
               offset: int,
               name='dstq'
               ):
    """Initialization of the DSTQ metric.

    Args:
      num_classes: Number of classes in the dataset as an integer.
      things_list: A sequence of class ids that belong to `things`.
      ignore_label: The class id to be ignored in evaluation as an integer or
        integer tensor.
      max_instances_per_category: The maximum number of instances for each class
        as an integer or integer tensor.
      offset: The maximum number of unique labels as an integer or integer
        tensor.
      name: An optional name. (default: 'dstq')
    """
    self._name = name
    self._num_classes = num_classes
    self._ignore_label = ignore_label
    self._things_list = things_list
    self._max_instances_per_category = max_instances_per_category

    if ignore_label >= num_classes:
      self._confusion_matrix_size = num_classes + 1
      self._include_indices = np.arange(self._num_classes)
    else:
      self._confusion_matrix_size = num_classes
      self._include_indices = np.array(
          [i for i in range(num_classes) if i != self._ignore_label])

    self._iou_confusion_matrix_per_sequence = collections.OrderedDict()
    self._predictions = collections.OrderedDict()
    self._ground_truth = collections.OrderedDict()
    self._intersections = collections.OrderedDict()
    self._sequence_length = collections.OrderedDict()
    self._depth_inlier = collections.OrderedDict()
    self._offset = offset
    lower_bound = num_classes * max_instances_per_category
    if offset < lower_bound:
      raise ValueError('The provided offset %d is too small. No guarantess '
                       'about the correctness of the results can be made. '
                       'Please choose an offset that is higher than num_classes'
                       ' * max_instances_per_category = %d' % lower_bound)

  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, depth_inlier: float,
                   sequence_id=0):
    """Accumulates the segmentation and tracking quality statistics.

    Args:
      y_true: The ground-truth panoptic label map for a particular video frame
        (defined as semantic_map * max_instances_per_category + instance_map).
      y_pred: The predicted panoptic label map for a particular video frame
        (defined as semantic_map * max_instances_per_category + instance_map).
      depth_inlier: The computed depth inlier ratio.
      sequence_id: The optional ID of the sequence the frames belong to. When no
        sequence is given, all frames are considered to belong to the same
        sequence (default: 0).
    """
    y_true = tf.cast(y_true, dtype=tf.int64)
    y_pred = tf.cast(y_pred, dtype=tf.int64)
    semantic_label = y_true // self._max_instances_per_category
    semantic_prediction = y_pred // self._max_instances_per_category
    # Check if the ignore value is outside the range [0, num_classes]. If yes,
    # map `_ignore_label` to `_num_classes`, so it can be used to create the
    # confusion matrix.
    if self._ignore_label > self._num_classes:
      semantic_label = tf.where(
          tf.not_equal(semantic_label, self._ignore_label), semantic_label,
          self._num_classes)
      semantic_prediction = tf.where(
          tf.not_equal(semantic_prediction, self._ignore_label),
          semantic_prediction, self._num_classes)
    if sequence_id in self._iou_confusion_matrix_per_sequence:
      self._iou_confusion_matrix_per_sequence[sequence_id] += (
          tf.math.confusion_matrix(
              tf.reshape(semantic_label, [-1]),
              tf.reshape(semantic_prediction, [-1]),
              self._confusion_matrix_size,
              dtype=tf.int64))
      self._sequence_length[sequence_id] += 1
    else:
      self._iou_confusion_matrix_per_sequence[sequence_id] = (
          tf.math.confusion_matrix(
              tf.reshape(semantic_label, [-1]),
              tf.reshape(semantic_prediction, [-1]),
              self._confusion_matrix_size,
              dtype=tf.int64))
      self._predictions[sequence_id] = {}
      self._ground_truth[sequence_id] = {}
      self._intersections[sequence_id] = {}
      self._sequence_length[sequence_id] = 1

    instance_label = y_true % self._max_instances_per_category

    label_mask = tf.zeros_like(semantic_label, dtype=tf.bool)
    prediction_mask = tf.zeros_like(semantic_prediction, dtype=tf.bool)
    for things_class_id in self._things_list:
      label_mask = tf.logical_or(label_mask,
                                 tf.equal(semantic_label, things_class_id))
      prediction_mask = tf.logical_or(
          prediction_mask, tf.equal(semantic_prediction, things_class_id))

    # Select the `crowd` region of the current class. This region is encoded
    # instance id `0`.
    is_crowd = tf.logical_and(tf.equal(instance_label, 0), label_mask)
    # Select the non-crowd region of the corresponding class as the `crowd`
    # region is ignored for the tracking term.
    label_mask = tf.logical_and(label_mask, tf.logical_not(is_crowd))
    # Do not punish id assignment for regions that are annotated as `crowd` in
    # the ground-truth.
    prediction_mask = tf.logical_and(prediction_mask, tf.logical_not(is_crowd))

    seq_preds = self._predictions[sequence_id]
    seq_gts = self._ground_truth[sequence_id]
    seq_intersects = self._intersections[sequence_id]

    # Compute and update areas of ground-truth, predictions and intersections.
    _update_dict_stats(seq_preds, y_pred[prediction_mask])
    _update_dict_stats(seq_gts, y_true[label_mask])

    non_crowd_intersection = tf.logical_and(label_mask, prediction_mask)
    intersection_ids = (
        y_true[non_crowd_intersection] * self._offset +
        y_pred[non_crowd_intersection])
    _update_dict_stats(seq_intersects, intersection_ids)

    if sequence_id in self._depth_inlier:
        self._depth_inlier[sequence_id].append(depth_inlier)
    else:
        self._depth_inlier[sequence_id] = []

  def result(self) -> Dict[Text, Any]:
    """Computes the segmentation and tracking quality.

    Returns:
      A dictionary containing:
        - 'DSTQ': The total DSTQ score.
        - 'DSTQ_per_seq': A list of the DSTQ score per sequence.
        - 'Depth_Inlier': The total depth inlier ratio.
        - 'Depth_Inlier_per_seq': A list of the depth inlier per sequence.
        - 'STQ': The total STQ score.
        - 'AQ': The total association quality (AQ) score.
        - 'IoU': The total mean IoU.
        - 'STQ_per_seq': A list of the STQ score per sequence.
        - 'AQ_per_seq': A list of the AQ score per sequence.
        - 'IoU_per_seq': A list of mean IoU per sequence.
        - 'Id_per_seq': A list of sequence Ids to map list index to sequence.
        - 'Length_per_seq': A list of the length of each sequence.
    """
    # Compute association quality (AQ)
    num_tubes_per_seq = [0] * len(self._ground_truth)
    aq_per_seq = [0] * len(self._ground_truth)
    iou_per_seq = [0] * len(self._ground_truth)
    id_per_seq = [''] * len(self._ground_truth)

    for index, sequence_id in enumerate(self._ground_truth):
      outer_sum = 0.0
      predictions = self._predictions[sequence_id]
      ground_truth = self._ground_truth[sequence_id]
      intersections = self._intersections[sequence_id]
      num_tubes_per_seq[index] = len(ground_truth)
      id_per_seq[index] = sequence_id

      for gt_id, gt_size in ground_truth.items():
        inner_sum = 0.0
        for pr_id, pr_size in predictions.items():
          tpa_key = self._offset * gt_id + pr_id
          if tpa_key in intersections:
            tpa = intersections[tpa_key].numpy()
            fpa = pr_size.numpy() - tpa
            fna = gt_size.numpy() - tpa
            inner_sum += tpa * (tpa / (tpa + fpa + fna))

        outer_sum += 1.0 / gt_size.numpy() * inner_sum
      aq_per_seq[index] = outer_sum

    aq_mean = np.sum(aq_per_seq) / np.maximum(np.sum(num_tubes_per_seq), 1e-15)
    aq_per_seq = aq_per_seq / np.maximum(num_tubes_per_seq, 1e-15)

    # Compute IoU scores.
    # The rows correspond to ground-truth and the columns to predictions.
    # Remove fp from confusion matrix for the void/ignore class.
    total_confusion = np.zeros(
        (self._confusion_matrix_size, self._confusion_matrix_size),
        dtype=np.int64)
    for index, confusion in enumerate(
        self._iou_confusion_matrix_per_sequence.values()):
      confusion = confusion.numpy()
      removal_matrix = np.zeros_like(confusion)
      removal_matrix[self._include_indices, :] = 1.0
      confusion *= removal_matrix
      total_confusion += confusion

      # `intersections` corresponds to true positives.
      intersections = confusion.diagonal()
      fps = confusion.sum(axis=0) - intersections
      fns = confusion.sum(axis=1) - intersections
      unions = intersections + fps + fns

      num_classes = np.count_nonzero(unions)
      ious = (intersections.astype(np.double) /
              np.maximum(unions, 1e-15).astype(np.double))
      iou_per_seq[index] = np.sum(ious) / num_classes

    # `intersections` corresponds to true positives.
    intersections = total_confusion.diagonal()
    fps = total_confusion.sum(axis=0) - intersections
    fns = total_confusion.sum(axis=1) - intersections
    unions = intersections + fps + fns

    num_classes = np.count_nonzero(unions)
    ious = (intersections.astype(np.double) /
            np.maximum(unions, 1e-15).astype(np.double))
    iou_mean = np.sum(ious) / num_classes

    depth_inlier_per_seq = []
    depth_inlier = []
    for index, sequence_id in enumerate(self._depth_inlier):
        depth_inliers = self._depth_inlier[sequence_id]
        depth_inlier += depth_inliers
        depth_inlier_per_seq.append(sum(depth_inliers) / len(depth_inliers))

    depth_inlier = sum(depth_inlier) / len(depth_inlier)

    st_quality = np.sqrt(aq_mean * iou_mean)
    st_quality_per_seq = np.sqrt(aq_per_seq * iou_per_seq)

    dst_quality = np.power(aq_mean * iou_mean * depth_inlier, 1/3.0)
    dst_quality_per_seq = np.power(aq_per_seq * iou_per_seq * depth_inlier_per_seq, 1/3.0)

    return {'DSTQ': dst_quality,
            'DSTQ_per_seq': dst_quality_per_seq,
            'Depth_Inlier': depth_inlier,
            'Depth_Inlier_per_seq': depth_inlier_per_seq,
            'STQ': st_quality,
            'AQ': aq_mean,
            'IoU': float(iou_mean),
            'STQ_per_seq': st_quality_per_seq,
            'AQ_per_seq': aq_per_seq,
            'IoU_per_seq': iou_per_seq,
            'ID_per_seq': id_per_seq,
            'Length_per_seq': list(self._sequence_length.values()),
            }

  def reset_states(self):
    """Resets all states that accumulated data."""
    self._iou_confusion_matrix_per_sequence = collections.OrderedDict()
    self._predictions = collections.OrderedDict()
    self._ground_truth = collections.OrderedDict()
    self._intersections = collections.OrderedDict()
    self._sequence_length = collections.OrderedDict()
    self._depth_inlier = collections.OrderedDict()


def scan_sequence(seq_gt_dir, seq_depth_dir, seq_pred_dir, dstq_obj, seq_id):
  gt_names = os.scandir(seq_gt_dir)
  gt_names = [name.name for name in gt_names if 'gtFine_class' in name.name]
  gt_names = [os.path.join(seq_gt_dir, name) for name in gt_names]
  gt_names = sorted(gt_names)

  depth_gt_names = os.scandir(seq_gt_dir)
  depth_gt_names = [
      name.name for name in depth_gt_names if 'depth' in name.name]
  depth_gt_names = [os.path.join(seq_gt_dir, name) for name in depth_gt_names]
  depth_gt_names = sorted(depth_gt_names)

  depth_pred_names = os.scandir(seq_depth_dir)
  depth_pred_names = [name.name for name in depth_pred_names]
  depth_pred_names = [os.path.join(seq_depth_dir, name)
                      for name in depth_pred_names]
  depth_pred_names = sorted(depth_pred_names)

  pred_names = os.scandir(seq_pred_dir)
  pred_names = [os.path.join(seq_pred_dir, name.name) for name in pred_names]
  cat_pred_names = [name for name in pred_names if name.endswith('cat.png')]
  ins_pred_names = [name for name in pred_names if name.endswith('ins.png')]
  cat_pred_names = sorted(cat_pred_names)
  ins_pred_names = sorted(ins_pred_names)

  assert len(np.unique([len(cat_pred_names), len(ins_pred_names), len(gt_names), len(depth_pred_names), len(depth_gt_names)])) == 1

  data_count = 0
  for cat_pred_name, ins_pred_name, gt_name, depth_pred_name, depth_gt_name in zip(
          cat_pred_names, ins_pred_names, gt_names, depth_pred_names, depth_gt_names):
    cat_pred = np.array(Image.open(cat_pred_name)).astype(np.int32)
    ins_pred = np.array(Image.open(ins_pred_name)).astype(np.int32)
    pred = cat_pred * (2 ** 16) + ins_pred

    cat_gt = np.array(Image.open(gt_name)).astype(np.int32)
    gt_name = gt_name.replace('class', 'instance')
    ins_gt = np.array(Image.open(gt_name)).astype(np.int32)
    gt = cat_gt * (2 ** 16) + ins_gt

    depth_pred = np.array(Image.open(depth_pred_name))
    depth_gt = np.array(Image.open(depth_gt_name))

    valid_mask = cat_gt != 255
    gt = gt[valid_mask]
    pred = pred[valid_mask]

    depth_pred = depth_pred.astype(np.float32)
    depth_gt = depth_gt.astype(np.float32)
    depth_pred = depth_pred[None, :, :, None]
    depth_pred = tf.cast(depth_pred, tf.float32)
    depth_pred = tf.compat.v1.image.resize_bilinear(
        depth_pred, depth_gt.shape, align_corners=True)
    depth_pred = depth_pred.numpy()
    depth_pred = depth_pred[0, :, :, 0]
    valid_mask = depth_gt > 0
    depth_gt = depth_gt[valid_mask]
    depth_pred = depth_pred[valid_mask]

    depth_inlier = np.maximum(depth_gt / depth_pred, depth_pred / depth_gt) < 1.25
    depth_inlier = np.sum(depth_inlier) / depth_inlier.shape[0]

    dstq_obj.update_state(gt, pred, depth_inlier, seq_id)
    data_count += 1


def main():
  dstq_obj = DSTQuality(19, list(range(8)), 255, 2 ** 16, 2 ** 16 * 256)

  for seq_id in ['12', '13']:
    seq_gt_dir = os.path.join(gt_dir, seq_id)
    seq_depth_dir = os.path.join(depth_dir, seq_id)
    seq_pred_dir = os.path.join(pred_dir, seq_id)
    scan_sequence(seq_gt_dir, seq_depth_dir, seq_pred_dir, dstq_obj, int(seq_id))

  result = dstq_obj.result()
  print(result)

  scores = {}
  scores['DSTQ'] = float(result['DSTQ']) * 100
  with open(os.path.join(out_dir, 'scores.txt'), 'w') as fout:
    yaml.dump(scores, fout, default_flow_style=False)


if __name__ == '__main__':
  main()
