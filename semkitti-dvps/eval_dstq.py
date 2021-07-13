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
from typing import MutableMapping, Sequence, Dict, Text, Any, Tuple, List
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


class STQuality(object):
  """Metric class for the Segmentation and Tracking Quality (STQ).
  The metric computes the geometric mean of two terms.
  - Association Quality: This term measures the quality of the track ID
      assignment for `thing` classes. It is formulated as a weighted IoU
      measure.
  - Segmentation Quality: This term measures the semantic segmentation quality.
      The standard class IoU measure is used for this.
  Example usage:
  stq_obj = segmentation_tracking_quality.STQuality(num_classes, things_list,
    ignore_label, max_instances_per_category, offset)
  stq_obj.update_state(y_true_1, y_pred_1)
  stq_obj.update_state(y_true_2, y_pred_2)
  ...
  result = stq_obj.result().numpy()
  """

  def __init__(self,
               num_classes: int,
               things_list: Sequence[int],
               ignore_label: int,
               max_instances_per_category: int,
               offset: int,
               name='stq'
               ):
    """Initialization of the STQ metric.
    Args:
      num_classes: Number of classes in the dataset as an integer.
      things_list: A sequence of class ids that belong to `things`.
      ignore_label: The class id to be ignored in evaluation as an integer or
        integer tensor.
      max_instances_per_category: The maximum number of instances for each class
        as an integer or integer tensor.
      offset: The maximum number of unique labels as an integer or integer
        tensor.
      name: An optional name. (default: 'st_quality')
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
    self._offset = offset
    lower_bound = num_classes * max_instances_per_category
    if offset < lower_bound:
      raise ValueError('The provided offset %d is too small. No guarantess '
                       'about the correctness of the results can be made. '
                       'Please choose an offset that is higher than num_classes'
                       ' * max_instances_per_category = %d' % lower_bound)

  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                   sequence_id=0):
    """Accumulates the segmentation and tracking quality statistics.
    Args:
      y_true: The ground-truth panoptic label map for a particular video frame
        (defined as semantic_map * max_instances_per_category + instance_map).
      y_pred: The predicted panoptic label map for a particular video frame
        (defined as semantic_map * max_instances_per_category + instance_map).
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

  def result(self) -> Dict[Text, Any]:
    """Computes the segmentation and tracking quality.
    Returns:
      A dictionary containing:
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

    st_quality = np.sqrt(aq_mean * iou_mean)
    st_quality_per_seq = np.sqrt(aq_per_seq * iou_per_seq)
    return {'STQ': st_quality,
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


class DSTQuality(STQuality):
  """Metric class for Depth-aware Segmentation and Tracking Quality (DSTQ).
  This metric computes STQ and the inlier depth metric (or depth quality (DQ))
  under several thresholds. Then it returns the geometric mean of DQ's, AQ and
  IoU to get the final DSTQ, i.e.,
  DSTQ@{threshold_1} = pow(STQ ** 2 * DQ@{threshold_1}, 1/3)
  DSTQ@{threshold_2} = pow(STQ ** 2 * DQ@{threshold_2}, 1/3)
  ...
  DSTQ = pow(STQ ** 2 * DQ, 1/3)
  where DQ = pow(prod_i^n(threshold_i), 1/n) for n depth thresholds.
  The default choices for depth thresholds are 1.1 and 1.25, i.e.,
  max(pred/gt, gt/pred) <= 1.1 and max(pred/gt, gt/pred) <= 1.25.
  Commonly used thresholds for the inlier metrics are 1.25, 1.25**2, 1.25**3.
  These thresholds are so loose that many methods achieves > 99%.
  Therefore, we choose 1.25 and 1.1 to encourage high-precision predictions.
  Example usage:
  dstq_obj = depth_aware_segmentation_and_tracking_quality.DSTQuality(
    num_classes, things_list, ignore_label, max_instances_per_category,
    offset, depth_threshold)
  dstq.update_state(y_true_1, y_pred_1, d_true_1, d_pred_1)
  dstq.update_state(y_true_2, y_pred_2, d_true_2, d_pred_2)
  ...
  result = dstq_obj.result().numpy()
  """

  _depth_threshold: Tuple[float, float] = (1.25, 1.1)
  _depth_total_counts: collections.OrderedDict
  _depth_inlier_counts: List[collections.OrderedDict]

  def __init__(self,
               num_classes: int,
               things_list: Sequence[int],
               ignore_label: int,
               max_instances_per_category: int,
               offset: int,
               depth_threshold: Tuple[float] = (1.25, 1.1),
               name: str = 'dstq',):  # pytype: disable=annotation-type-mismatch
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
      depth_threshold: A sequence of depth thresholds for the depth quality.
        (default: (1.25, 1.1))
      name: An optional name. (default: 'dstq')
    """
    super().__init__(num_classes, things_list, ignore_label,
                     max_instances_per_category, offset, name)
    if not (isinstance(depth_threshold, tuple) or
            isinstance(depth_threshold, list)):
      raise TypeError('The type of depth_threshold must be tuple or list.')
    if not depth_threshold:
      raise ValueError('depth_threshold must be non-empty.')
    self._depth_threshold = tuple(depth_threshold)
    self._depth_total_counts = collections.OrderedDict()
    self._depth_inlier_counts = []
    for _ in range(len(self._depth_threshold)):
      self._depth_inlier_counts.append(collections.OrderedDict())

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   d_true: tf.Tensor,
                   d_pred: tf.Tensor,
                   sequence_id: int = 0):
    """Accumulates the depth-aware segmentation and tracking quality statistics.
    Args:
      y_true: The ground-truth panoptic label map for a particular video frame
        (defined as semantic_map * max_instances_per_category + instance_map).
      y_pred: The predicted panoptic label map for a particular video frame
        (defined as semantic_map * max_instances_per_category + instance_map).
      d_true: The ground-truth depth map for this video frame.
      d_pred: The predicted depth map for this video frame.
      sequence_id: The optional ID of the sequence the frames belong to. When no
        sequence is given, all frames are considered to belong to the same
        sequence (default: 0).
    """
    super().update_state(y_true, y_pred, sequence_id)
    # Valid depth labels contain positive values.
    d_valid_mask = d_true > 0
    d_valid_total = tf.reduce_sum(tf.cast(d_valid_mask, tf.int32))
    # Valid depth prediction is expected to contain positive values.
    d_valid_mask = tf.logical_and(d_valid_mask, d_pred > 0)
    d_valid_true = tf.boolean_mask(d_true, d_valid_mask)
    d_valid_pred = tf.boolean_mask(d_pred, d_valid_mask)
    inlier_error = tf.maximum(d_valid_pred / d_valid_true,
                              d_valid_true / d_valid_pred)
    # For each threshold, count the number of inliers.
    for threshold_index, threshold in enumerate(self._depth_threshold):
      num_inliers = tf.reduce_sum(tf.cast(inlier_error <= threshold, tf.int32))
      inlier_counts = self._depth_inlier_counts[threshold_index]
      inlier_counts[sequence_id] = (inlier_counts.get(sequence_id, 0) +
                                    int(num_inliers.numpy()))
    # Update the total counts of the depth labels.
    self._depth_total_counts[sequence_id] = (
        self._depth_total_counts.get(sequence_id, 0) +
        int(d_valid_total.numpy()))

  def result(self):
    """Computes the depth-aware segmentation and tracking quality.
    Returns:
      A dictionary containing:
        - 'STQ': The total STQ score.
        - 'AQ': The total association quality (AQ) score.
        - 'IoU': The total mean IoU.
        - 'STQ_per_seq': A list of the STQ score per sequence.
        - 'AQ_per_seq': A list of the AQ score per sequence.
        - 'IoU_per_seq': A list of mean IoU per sequence.
        - 'Id_per_seq': A list of sequence Ids to map list index to sequence.
        - 'Length_per_seq': A list of the length of each sequence.
        - 'DSTQ': The total DSTQ score.
        - 'DSTQ@thres': The total DSTQ score for threshold thres
        - 'DSTQ_per_seq@thres': A list of DSTQ score per sequence for thres.
        - 'DQ': The total DQ score.
        - 'DQ@thres': The total DQ score for threshold thres.
        - 'DQ_per_seq@thres': A list of DQ score per sequence for thres.
    """
    # Gather the results for STQ.
    stq_results = super().result()
    # Collect results for depth quality per sequecne and threshold.
    dq_per_seq_at_threshold = {}
    dq_at_threshold = {}
    for threshold_index, threshold in enumerate(self._depth_threshold):
      dq_per_seq_at_threshold[threshold] = [0] * len(self._ground_truth)
      total_count = 0
      inlier_count = 0
      # Follow the order of computing STQ by enumerating _ground_truth.
      for index, sequence_id in enumerate(self._ground_truth):
        sequence_inlier = self._depth_inlier_counts[threshold_index][
            sequence_id]
        sequence_total = self._depth_total_counts[sequence_id]
        if sequence_total > 0:
          dq_per_seq_at_threshold[threshold][
              index] = sequence_inlier / sequence_total
        total_count += sequence_total
        inlier_count += sequence_inlier
      if total_count == 0:
        dq_at_threshold[threshold] = 0
      else:
        dq_at_threshold[threshold] = inlier_count / total_count
    # Compute DQ as the geometric mean of DQ's at different thresholds.
    dq = 1
    for _, threshold in enumerate(self._depth_threshold):
      dq *= dq_at_threshold[threshold]
    dq = dq ** (1 / len(self._depth_threshold))
    dq_results = {}
    dq_results['DQ'] = dq
    for _, threshold in enumerate(self._depth_threshold):
      dq_results['DQ@{}'.format(threshold)] = dq_at_threshold[threshold]
      dq_results['DQ_per_seq@{}'.format(
          threshold)] = dq_per_seq_at_threshold[threshold]
    # Combine STQ and DQ to get DSTQ.
    dstq_results = {}
    dstq_results['DSTQ'] = (stq_results['STQ'] ** 2 * dq) ** (1/3)
    for _, threshold in enumerate(self._depth_threshold):
      dstq_results['DSTQ@{}'.format(threshold)] = (
          stq_results['STQ'] ** 2 * dq_at_threshold[threshold]) ** (1/3)
      dstq_results['DSTQ_per_seq@{}'.format(threshold)] = [
          (stq_result**2 * dq_result)**(1 / 3) for stq_result, dq_result in zip(
              stq_results['STQ_per_seq'], dq_per_seq_at_threshold[threshold])
      ]
    # Merge all the results.
    dstq_results.update(stq_results)
    dstq_results.update(dq_results)
    return dstq_results

  def reset_states(self):
    """Resets all states that accumulated data."""
    super().reset_states()
    self._depth_total_counts = collections.OrderedDict()
    self._depth_inlier_counts = []
    for _ in range(len(self._depth_threshold)):
      self._depth_inlier_counts.append(collections.OrderedDict())


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

  assert len(np.unique([len(cat_pred_names), len(ins_pred_names), len(gt_names),
      len(depth_pred_names), len(depth_gt_names)])) == 1

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

    dstq_obj.update_state(gt, pred, depth_gt, depth_pred, seq_id)


def main():
  dstq_obj = DSTQuality(19, list(range(8)), 255, 2 ** 16, 2 ** 16 * 256, [1.25])

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


main()
