# Copyright 2017 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Additional Numpy methods. Big mess of many things!
"""
import numpy as np


# =========================================================================== #
# Numpy implementations of SSD boxes functions.
# =========================================================================== #
def ssd_bboxes_decode(feat_localizations,  # first->[N,H,W,anchorNum,4]                            
                      anchor_bboxes,                # first->[(N),(38*38*1),(38*38*1),(4*1),(4*1)]
                      prior_scaling=[0.1, 0.1, 0.2, 0.2]): # variance超参数来调整检测值
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Return:
      numpy array Nx4: ymin, xmin, ymax, xmax
    """
    # Reshape for easier broadcasting.
    l_shape = feat_localizations.shape
    feat_localizations = np.reshape(feat_localizations,  # [N*H*W,anchorNum,4]
                                    (-1, l_shape[-2], l_shape[-1]))
    yref, xref, href, wref = anchor_bboxes # first->[(N),(38*38*1),(38*38*1),(4*1),(4*1)]
    xref = np.reshape(xref, [-1, 1]) # first->[38*38,1]
    yref = np.reshape(yref, [-1, 1]) # first->[38*38,1]

    # Compute center, height and width 相当于解码过程，通过偏移量还原坐标
    # 参考：https://www.codetd.com/article/1534601
    cx = feat_localizations[:, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, 1] * href * prior_scaling[1] + yref
    w = wref * np.exp(feat_localizations[:, :, 2] * prior_scaling[2])
    h = href * np.exp(feat_localizations[:, :, 3] * prior_scaling[3])
    # bboxes: ymin, xmin, xmax, ymax.
    bboxes = np.zeros_like(feat_localizations) #构造一个矩阵，其维度与矩阵feat_localizations一致，并为其初始化为全0
    # 坐标形式由[cx,cy,w,h]变为[xmin,ymin,xmax,ymax]
    bboxes[:, :, 0] = cy - h / 2.
    bboxes[:, :, 1] = cx - w / 2.
    bboxes[:, :, 2] = cy + h / 2.
    bboxes[:, :, 3] = cx + w / 2.
    # Back to original shape.
    bboxes = np.reshape(bboxes, l_shape)
    return bboxes


def ssd_bboxes_select_layer(predictions_layer, # first->[N,H,W,anchorNum,21]
                            localizations_layer,      # first->[N,H,W,anchorNum,4]
                            anchors_layer,            # first->[((N),(38*38*1),(38*38*1),(4*1),(4*1))]
                            select_threshold=0.5,
                            img_shape=(300, 300),
                            num_classes=21,
                            decode=True):
    """Extract classes, scores and bounding boxes from features in one layer.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    # First decode localizations features if necessary.
    if decode:
        localizations_layer = ssd_bboxes_decode(localizations_layer, anchors_layer)

    # Reshape features to: Batches x N x N_labels | 4.
    p_shape = predictions_layer.shape
    batch_size = p_shape[0] if len(p_shape) == 5 else 1
    predictions_layer = np.reshape(predictions_layer,     #[N,H*W*anchorNum,21]
                                   (batch_size, -1, p_shape[-1]))
    l_shape = localizations_layer.shape
    localizations_layer = np.reshape(localizations_layer, #[N,H*W*anchorNum,4]
                                     (batch_size, -1, l_shape[-1]))

    # Boxes selection: use threshold or score > no-label criteria.
    if select_threshold is None or select_threshold == 0:
        # Class prediction and scores: assign 0. to 0-class
        classes = np.argmax(predictions_layer, axis=2)
        scores = np.amax(predictions_layer, axis=2)
        mask = (classes > 0)
        classes = classes[mask]
        scores = scores[mask]
        bboxes = localizations_layer[mask]
    else: 
        #假设[1,5*5*4,20]的sub_predictions，表示我们从1张图片中所有的点（5*5*4）中找到类别得分大于阈值的点对应的类别，
        #因为这里阈值是0.5，所以对于一个点只会对应一种类别
        sub_predictions = predictions_layer[:, :, 1:] # 去掉背景[N,H*W*anchorNum,20]
        idxes = np.where(sub_predictions > select_threshold) #得分大与阈值的坐标[x,y,z]->[batch,H*W*anchorNum,类别]
        classes = idxes[-1]+1  #可能的类别名（这里类别名是数字，因为第一个类别为背景，所以下标要加1）
        scores = sub_predictions[idxes]  #大于阈值的类别得分[batch,H*W*anchorNum,大于阈值的类别]
        bboxes = localizations_layer[idxes[:-1]]#大于阈值的类别位置[batch,H*W*anchorNum,大于阈值的类别的位置]
        #参看：https://drive.google.com/file/d/164mVbMBhoMzY5pkaEOdK3IIcIwTOj2B-/view
    return classes, scores, bboxes


def ssd_bboxes_select(predictions_net,  #预测概率[N,H,W,anchorNum,4,6]
                      localizations_net,       #预测位置[N,H,W,anchorNum,21,6]
                      anchors_net,             #anchor[((N),(38*38*1),(38*38*1),(4*1),(4*1)),6]
                      select_threshold=0.5,
                      img_shape=(300, 300),
                      num_classes=21,
                      decode=True):
    """Extract classes, scores and bounding boxes from network output layers.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    l_classes = []
    l_scores = []
    l_bboxes = []
    # l_layers = []
    # l_idxes = []
    for i in range(len(predictions_net)):  # 6个feature map
        #对于每一个feature map上所有的框，每一个框都会有C个类别的得分，然后对每一个框进行阈值（类别得分阈值）判断
        #保留大于阈值得分的框，低于阈值的框舍弃
        classes, scores, bboxes = ssd_bboxes_select_layer(
            predictions_net[i], localizations_net[i], anchors_net[i],
            select_threshold, img_shape, num_classes, decode)
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)
        # Debug information.
        # l_layers.append(i)
        # l_idxes.append((i, idxes))

    #把6个feature map层找到的所有点拼接到一个列表里
    classes = np.concatenate(l_classes, 0)
    scores = np.concatenate(l_scores, 0)
    bboxes = np.concatenate(l_bboxes, 0)
    return classes, scores, bboxes


# =========================================================================== #
# Common functions for bboxes handling and selection.
# =========================================================================== #
#保留指定数量的边界框
#得到了裁剪之后的边界框和类别，我们需要对其做一个降序排序，并保留指定数量的最优的那一部分。
#个中参数可以自行调节，这里只保留至多400个。
def bboxes_sort(classes, scores, bboxes, top_k=400):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    # if priority_inside:
    #     inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
    #         (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
    #     idxes = np.argsort(-scores)
    #     inside = inside[idxes]
    #     idxes = np.concatenate([idxes[inside], idxes[~inside]])
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes

#边界框的裁剪
#此时得到的rbboxes数组中的部分值是大于1的，而在我们的归一化表示中，整个框的宽度和高度都是等于1的，
#因此需要对其进行裁剪，保证最大值不超过1，最小值不小于0。
def bboxes_clip(bbox_ref, bboxes): #假设对于一张图片bbox_ref->[0,0,1,1],bboxes->[(33*4)]
    """Clip bounding boxes with respect to reference bbox.
    """
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)     #（4*33）
    bbox_ref = np.transpose(bbox_ref) #（4*1）
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes

#对bboxes进行resize，使其恢复到original image shape。但是这里使用的是归一化的表示方式
def bboxes_resize(bbox_ref, bboxes):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform.
    """
    bboxes = np.copy(bboxes)
    # Translate.
    bboxes[:, 0] -= bbox_ref[0]
    bboxes[:, 1] -= bbox_ref[1]
    bboxes[:, 2] -= bbox_ref[0]
    bboxes[:, 3] -= bbox_ref[1]
    # Resize.
    resize = [bbox_ref[2] - bbox_ref[0], bbox_ref[3] - bbox_ref[1]]
    bboxes[:, 0] /= resize[0]
    bboxes[:, 1] /= resize[1]
    bboxes[:, 2] /= resize[0]
    bboxes[:, 3] /= resize[1]
    return bboxes

# 计算IOU
def bboxes_jaccard(bboxes1, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard


def bboxes_intersection(bboxes_ref, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes_ref = np.transpose(bboxes_ref)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes_ref[0], bboxes2[0])
    int_xmin = np.maximum(bboxes_ref[1], bboxes2[1])
    int_ymax = np.minimum(bboxes_ref[2], bboxes2[2])
    int_xmax = np.minimum(bboxes_ref[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol = (bboxes_ref[2] - bboxes_ref[0]) * (bboxes_ref[3] - bboxes_ref[1])
    score = int_vol / vol
    return score

#在得到了指定数量的边界框和类别之后。对于同一个类存在多个框的情况下，
#要找到一个最合适的，并去掉其他冗余的框，需要进行非极大值抑制的操作。
def bboxes_nms(classes, scores, bboxes, nms_threshold=0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    # 边框保留标志
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1): #对于所有得分大于阈值的框
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):]) #计算当前框与之后所有框的IOU
            # Overlap threshold for keeping + checking part of the same class
            # 对于所有IOU小于0.45或者当前类别与之后类别不相同的位置，置为True
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            # 将上面得到的所有IOU小于0.45或类别不同的位置赋给keep_bboxes
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]


def bboxes_nms_fast(classes, scores, bboxes, threshold=0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    pass




