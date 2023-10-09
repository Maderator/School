#!/usr/bin/env python3

# edbe2dad-018e-11eb-9574-ea7484399335
# 44752d3d-fdd8-11ea-9574-ea7484399335

import numpy as np
import tensorflow as tf

BACKEND = tf # or you can use `tf` for TensorFlow implementation

TOP, LEFT, BOTTOM, RIGHT = range(4)

def bboxes_area(bboxes):
    """ Compute area of given set of bboxes.

    The computation can be performed either using Numpy or TensorFlow.
    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    If the bboxes.shape is [..., 4], the output shape is bboxes.shape[:-1].
    """
    return BACKEND.maximum(bboxes[..., BOTTOM] - bboxes[..., TOP], 0) \
        * BACKEND.maximum(bboxes[..., RIGHT] - bboxes[..., LEFT], 0)

def bboxes_iou(xs, ys):
    """ Compute IoU of corresponding pairs from two sets of bboxes xs and ys.

    The computation can be performed either using Numpy or TensorFlow.
    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    Note that broadcasting is supported, so passing inputs with
    xs.shape=[num_xs, 1, 4] and ys.shape=[1, num_ys, 4] will produce output
    with shape [num_xs, num_ys], computing IoU for all pairs of bboxes from
    xs and ys. Formally, the output shape is np.broadcast(xs, ys).shape[:-1].
    """
    intersections = BACKEND.stack([
        BACKEND.maximum(xs[..., TOP], ys[..., TOP]),
        BACKEND.maximum(xs[..., LEFT], ys[..., LEFT]),
        BACKEND.minimum(xs[..., BOTTOM], ys[..., BOTTOM]),
        BACKEND.minimum(xs[..., RIGHT], ys[..., RIGHT]),
    ], axis=-1)

    xs_area, ys_area, intersections_area = bboxes_area(xs), bboxes_area(ys), bboxes_area(intersections)

    return intersections_area / (xs_area + ys_area - intersections_area)

def bboxes_to_fast_rcnn(anchors, bboxes):
    """ Convert `bboxes` to a Fast-R-CNN-like representation relative to `anchors`.

    The `anchors` and `bboxes` are arrays of four-tuples (top, left, bottom, right);
    you can use the TOP, LEFT, BOTTOM, RIGHT constants as indices of the
    respective coordinates.

    The resulting representation of a single bbox is a four-tuple with:
    - (bbox_y_center - anchor_y_center) / anchor_height
    - (bbox_x_center - anchor_x_center) / anchor_width
    - log(bbox_height / anchor_height)
    - log(bbox_width / anchor_width)

    If the anchors.shape is [anchors_len, 4], bboxes.shape is [anchors_len, 4],
    the output shape is [anchors_len, 4].
    """

    anchors_height = anchors[..., BOTTOM] - anchors[..., TOP] 
    anchors_width = anchors[..., RIGHT] - anchors[..., LEFT]
    anchors_cx = (anchors[..., LEFT] + anchors[..., RIGHT]) / 2.0
    anchors_cy = (anchors[..., TOP] + anchors[..., BOTTOM]) / 2.0
    
    bboxes_height = bboxes[..., BOTTOM] - bboxes[..., TOP]
    bboxes_width = bboxes[..., RIGHT] - bboxes[..., LEFT]
    bboxes_cx = (bboxes[..., LEFT] + bboxes[..., RIGHT]) / 2.0
    bboxes_cy = (bboxes[..., TOP] + bboxes[..., BOTTOM]) / 2.0
    fr_y = (bboxes_cy - anchors_cy) / anchors_height
    fr_x = (bboxes_cx - anchors_cx) / anchors_width
    fr_h = tf.math.log(bboxes_height / anchors_height) 
    fr_w = tf.math.log(bboxes_width / anchors_width)

    return tf.stack([fr_y, fr_x, fr_h, fr_w], axis=1)

def bboxes_from_fast_rcnn(anchors, fast_rcnns):
    """ Convert Fast-R-CNN-like representation relative to `anchor` to a `bbox`.

    The anchors.shape is [anchors_len, 4], fast_rcnns.shape is [anchors_len, 4],
    the output shape is [anchors_len, 4].
    """
    anchors_height = anchors[..., BOTTOM] - anchors[..., TOP]
    anchors_width = anchors[..., RIGHT] - anchors[..., LEFT]
    anchors_cx = (anchors[..., LEFT] + anchors[..., RIGHT]) / 2.0
    anchors_cy = (anchors[..., TOP] + anchors[..., BOTTOM]) / 2.0
    
    fr_y = fast_rcnns[..., 0]
    fr_x = fast_rcnns[..., 1]
    fr_h = fast_rcnns[..., 2]
    fr_w = fast_rcnns[..., 3]

    bboxes_height = tf.math.exp(fr_h) * anchors_height
    bboxes_width = tf.math.exp(fr_w) * anchors_width
    bboxes_cx = fr_x * anchors_width + anchors_cx
    bboxes_cy = fr_y * anchors_height + anchors_cy

    bboxes_top = bboxes_cy - bboxes_height / 2.0
    bboxes_left = bboxes_cx - bboxes_width / 2.0
    bboxes_bottom = bboxes_top + bboxes_height
    bboxes_right = bboxes_left + bboxes_width
    return tf.stack([bboxes_top, bboxes_left, bboxes_bottom, bboxes_right], axis=1)

def bboxes_training(anchors, gold_classes, gold_bboxes, iou_threshold):
    """ Compute training data for object detection.

    Arguments:
    - `anchors` is an array of four-tuples (top, left, bottom, right)
    - `gold_classes` is an array of zero-based classes of the gold objects
    - `gold_bboxes` is an array of four-tuples (top, left, bottom, right)
      of the gold objects
    - `iou_threshold` is a given threshold

    Returns:
    - `anchor_classes` contains for every anchor either 0 for background
      (if no gold object is assigned) or `1 + gold_class` if a gold object
      with `gold_class` is assigned to it
    - `anchor_bboxes` contains for every anchor a four-tuple
      `(center_y, center_x, height, width)` representing the gold bbox of
      a chosen object using parametrization of Fast R-CNN; zeros if no
      gold object was assigned to the anchor

    Algorithm:
    - First, for each gold object, assign it to an anchor with the largest IoU
      (the one with smaller index if there are several). In case several gold
      objects are assigned to a single anchor, use the gold object with smaller
      index.
    - For each unused anchors, find the gold object with the largest IoU
      (again the one with smaller index if there are several), and if the IoU
      is >= iou_threshold, assign the object to the anchor.
    """

    # TODO: First, for each gold object, assign it to an anchor with the
    # largest IoU (the one with smaller index if there are several). In case
    # several gold objects are assigned to a single anchor, use the gold object
    # with smaller index.
    anch_gold_iou = bboxes_iou(tf.expand_dims(anchors,axis=1), tf.expand_dims(gold_bboxes,axis=0))
    
    # assign max_iou_anchors indices to gold_bboxes
    anch_indices_max_iou = tf.math.argmax(anch_gold_iou, axis=0)
    # one_hot representation of anchors indices
    anch_oh = tf.one_hot(anch_indices_max_iou, depth=len(anchors), axis=0)
    # get indices of chosen objects for each anchor
    anchor_chosen_gobject = tf.math.argmax(anch_oh, axis=1)
    # get one_hot representation of possible first gold object with smallest index.
    possible_first_gobject_oh = tf.one_hot(anchor_chosen_gobject, depth=len(gold_classes), axis=1)
    # Do a logical_and of anch_oh and possible_first_gobject_oh to get mask of the first gold object for each anchor or row with all 
    # indices 0.
    mask_g_objects_oh = tf.math.logical_and(tf.cast(anch_oh, dtype=tf.bool), tf.cast(possible_first_gobject_oh, dtype=tf.bool))
    mask_g_objects = tf.math.reduce_any(mask_g_objects_oh, axis=0)
    # use the mask on anch_indices
    anch_indices_max_iou *= tf.cast(mask_g_objects, tf.int64)

    # TODO: For each unused anchors, find the gold object with the largest IoU
    # (again the one with smaller index if there are several), and if the IoU
    # is >= threshold, assign the object to the anchor.
    anch_oh = tf.cast(mask_g_objects_oh, dtype=tf.float32)
    anch_gold_iou_thresholded = anch_gold_iou * tf.cast(anch_gold_iou >= iou_threshold, tf.float32) # values < iou_threshold will be zero after this
    anch_oh = anch_gold_iou_thresholded + anch_oh * 2.0
    anch_oh = tf.pad(anch_oh, [[0,0],[1,0]], mode="CONSTANT", constant_values=0)

    gold_classes_extended = tf.concat([[0], gold_classes+1], axis=0)
    cl_indices = tf.math.argmax(anch_oh, axis=1)

    anchor_classes = tf.gather(gold_classes_extended, cl_indices)

    gold_bboxes_extended = tf.concat([[(0.0,0.0,0.0,0.0)], gold_bboxes], axis=0)

    anchor_bboxes = tf.gather(gold_bboxes_extended, cl_indices)

    anchor_bboxes = bboxes_to_fast_rcnn(anchors, anchor_bboxes)

    # Return zero if no golden object was assigned to the anchor    
    rg = tf.constant(list(range(len(cl_indices))))
    nonzero = tf.cast(cl_indices > 0, tf.int32)
    nonzero_cl_indices = tf.stack([nonzero,rg], axis=1)

    zeros = tf.tile([(0.0,0.0,0.0,0.0)], [len(nonzero_cl_indices),1])
    anchor_bboxes = tf.stack(
                        [zeros,anchor_bboxes],
                        axis=0
                    )
    anchor_bboxes = tf.gather_nd(anchor_bboxes, nonzero_cl_indices)
    
    return anchor_classes, anchor_bboxes

def main(args):
    return bboxes_to_fast_rcnn, bboxes_from_fast_rcnn, bboxes_training

import unittest
class Tests(unittest.TestCase):
    def test_bboxes_to_from_fast_rcnn(self):
        data = [
            [[0, 0, 10, 10], [0, 0, 10, 10], [0,  0, 0, 0]],
            [[0, 0, 10, 10], [5, 0, 15, 10], [.5, 0, 0, 0]],
            [[0, 0, 10, 10], [0, 5, 10, 15], [0, .5, 0, 0]],
            [[0, 0, 10, 10], [0, 0, 20, 30], [.5, 1, np.log(2), np.log(3)]],
            [[0, 9, 10, 19], [2, 10, 5, 16], [-0.15, -0.1, -1.2039728, -0.5108256]],
            [[5, 3, 15, 13], [7, 7, 10, 9], [-0.15, 0, -1.2039728, -1.609438]],
            [[7, 6, 17, 16], [9, 10, 12, 13], [-0.15, 0.05, -1.2039728, -1.2039728]],
            [[5, 6, 15, 16], [7, 7, 10, 10], [-0.15, -0.25, -1.2039728, -1.2039728]],
            [[6, 3, 16, 13], [8, 5, 12, 8], [-0.1, -0.15, -0.9162907, -1.2039728]],
            [[5, 2, 15, 12], [9, 6, 12, 8], [0.05, 0, -1.2039728, -1.609438]],
            [[2, 10, 12, 20], [6, 11, 8, 17], [0, -0.1, -1.609438, -0.5108256]],
            [[10, 9, 20, 19], [12, 13, 17, 16], [-0.05, 0.05, -0.6931472, -1.2039728]],
            [[6, 7, 16, 17], [10, 11, 12, 14], [0, 0.05, -1.609438, -1.2039728]],
            [[2, 2, 12, 12], [3, 5, 8, 8], [-0.15, -0.05, -0.6931472, -1.2039728]],
        ]
        # First run on individual anchors, and then on all together
        for anchors, bboxes, fast_rcnns in [[[anchor], [bbox], [fast_rcnn]] for anchor, bbox, fast_rcnn in data] + [zip(*data)]:
            anchors, bboxes, fast_rcnns = [np.array(data, np.float32) for data in [anchors, bboxes, fast_rcnns]]
            np.testing.assert_almost_equal(bboxes_to_fast_rcnn(anchors, bboxes), fast_rcnns, decimal=3)
            np.testing.assert_almost_equal(bboxes_from_fast_rcnn(anchors, fast_rcnns), bboxes, decimal=3)

    def test_bboxes_training(self):
        anchors = np.array([[0, 0, 10, 10], [0, 10, 10, 20], [10, 0, 20, 10], [10, 10, 20, 20]], np.float32)
        for gold_classes, gold_bboxes, anchor_classes, anchor_bboxes, iou in [
                [[1], [[14., 14, 16, 16]], [0, 0, 0, 2], [[0, 0, 0, 0]] * 3 + [[0, 0, np.log(1/5), np.log(1/5)]], 0.5],
                [[2], [[0., 0, 20, 20]], [3, 0, 0, 0], [[.5, .5, np.log(2), np.log(2)]] + [[0, 0, 0, 0]] * 3, 0.26],
                [[2], [[0., 0, 20, 20]], [3, 3, 3, 3], [[y, x, np.log(2), np.log(2)] for y in [.5, -.5] for x in [.5, -.5]], 0.24],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 0, 1],
                 [[0, 0, 0 ,0], [0, 0, 0, 0], [0, 0, 0, 0], [-0.35, -0.45, 0.53062826, 0.4054651]], 0.5],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 2, 1],
                 [[0, 0, 0 ,0], [0, 0, 0, 0], [-0.1, 0.6, -0.22314353, 0.6931472], [-0.35, -0.45, 0.53062826, 0.4054651]], 0.3],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 1, 2, 1],
                 [[0, 0, 0 ,0], [0.65, -0.45, 0.53062826, 0.4054651], [-0.1, 0.6, -0.22314353, 0.6931472], [-0.35, -0.45, 0.53062826, 0.4054651]], 0.17],
        ]:
            gold_classes, anchor_classes = np.array(gold_classes, np.int32), np.array(anchor_classes, np.int32)
            gold_bboxes, anchor_bboxes = np.array(gold_bboxes, np.float32), np.array(anchor_bboxes, np.float32)
            computed_classes, computed_bboxes = bboxes_training(anchors, gold_classes, gold_bboxes, iou)
            np.testing.assert_almost_equal(computed_classes, anchor_classes, decimal=3)
            np.testing.assert_almost_equal(computed_bboxes, anchor_bboxes, decimal=3)

if __name__ == '__main__':
    unittest.main()
