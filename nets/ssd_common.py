# Copyright 2015 Paul Balanca. All Rights Reserved.
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
"""Shared function between different SSD implementations.
"""
import numpy as np
import tensorflow as tf
import tf_extended as tfe


# =========================================================================== #
# TensorFlow implementation of boxes SSD encoding / decoding.
# =========================================================================== #
def tf_ssd_bboxes_encode_layer(labels, #gt标签，1D的tensor,shape(m,) 
                               bboxes,              #Nx4的Tensor(float)，真实的bbox,shape(m,4) 
                               anchors_layer,       #参考锚点list[(38,38,1), (38,38,1), (4), (4)]
                               num_classes,         #分类类别数21
                               no_annotation_label,
                               ignore_threshold=0.5, #gt和锚点框间的匹配阈值，大于该值则为正样本
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],#真实值到预测值转换中用到的缩放
                               dtype=tf.float32):
    """Encode groundtruth labels and bounding boxes using SSD anchors from
    one layer.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return: 返回：包含目标标签类别，目标位置，目标置信度的tesndor 
      (target_labels, target_localizations, target_scores): Target Tensors.
    """
    # Anchors coordinates and volume.
    yref, xref, href, wref = anchors_layer#此前每个特征图上点对应生成的锚点框作为参考框
    ymin = yref - href / 2.   #求参考框的左上角点（xmin,ymin）和右下角点(xmax,ymax) 
    xmin = xref - wref / 2.   #yref和xref的shape为（38,38,1）；href和wref的shape为（4，）
    ymax = yref + href / 2.   #这里计算时会把（38,38,1）扩展成（38,38,4）
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)  #求参考框面积vol_anchors

    # Initialize tensors...    #shape表示每个特征图上总锚点数
    shape = (yref.shape[0], yref.shape[1], href.size) #对于第一个特征图，shape=(38,38,4)；第二个特征图的shape=(19,19,6)
    feat_labels = tf.zeros(shape, dtype=tf.int64) #初始化每个特征图上的点对应的各个box所属标签维度 如：38x38x4  
    feat_scores = tf.zeros(shape, dtype=dtype)    #初始化每个特征图上的点对应的各个box所属标目标的得分值维度 如：38x38x4 

    feat_ymin = tf.zeros(shape, dtype=dtype)  #预测每个特征图每个点所属目标的坐标 ；如38x38x4;初始化为全0
    feat_xmin = tf.zeros(shape, dtype=dtype)  
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    #根据默认框和ground truth box的jaccard 重叠来寻找对应的默认框。文章中选取了jaccard重叠超过0.5的默认框为正样本，其它为负样本。
    def jaccard_with_anchors(bbox):  #计算gt的框和参考锚点框的重合度
        """Compute jaccard score between a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0]) #计算重叠区域的坐标，feature map（38*38*4）和ground truth（4，）
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.) #计算重叠区域的长与宽  
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w         #重叠区域的面积
        union_vol = vol_anchors - inter_vol \   #计算bbox和参考框的并集区域
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol, union_vol)  #计算IOU并返回该值
        return jaccard

    def intersection_with_anchors(bbox):#计算某个参考框包含真实框的得分情况
        """Compute intersection between score a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])    #计算bbox和锚点框重叠区域的坐标和长宽 
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w    #重叠区域面积
        scores = tf.div(inter_vol, vol_anchors) #将重叠区域面积除以参考框面积作为该参考框得分值；
        return scores

    #条件函数（i<ground truth box数量）
    def condition(i, feat_labels, feat_scores,   
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(labels))   # 逐元素比较大小,遍历labels，因为i在body返回的时候加1了 
        return r[0]

    #计算所有ground truth与一个feature map上先验框的IOU(置信度)
    def body(i, feat_labels, feat_scores, #该函数大致意思是选择与gt box IOU最大的锚点框负责回归任务，并预测对应的边界框，如此循环  
             feat_ymin, feat_xmin, feat_ymax, feat_xmax): #shape=(38,38,4)
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.  #计算bbox与参考框的IOU值 
        label = labels[i] #第i个gt框类别
        bbox = bboxes[i]  #第i个gt框坐标
        jaccard = jaccard_with_anchors(bbox)
        # Mask: check threshold + scores + no annotations + num_classes.
        # feature map上的某个先验框可能与多个ground truth有交集，我们只保留IOU最大的
        mask = tf.greater(jaccard, feat_scores)   
        # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        mask = tf.logical_and(mask, feat_scores > -0.5)  '''--？？？？feat_scores > -0.5什么意思？？？？--'''
        mask = tf.logical_and(mask, label < num_classes) #label满足<21 
        imask = tf.cast(mask, tf.int64)  #将mask转换数据类型int型 
        fmask = tf.cast(mask, dtype)     #将mask转换数据类型float型
        # Update values using mask.
        feat_labels = imask * label + (1 - imask) * feat_labels #当mask=1，则feat_labels=1；否则为0，即背景（38，38，4）
        feat_scores = tf.where(mask, jaccard, feat_scores)      #tf.where表示如果mask为真则jaccard(置信度)，否则为feat_scores(默认0)

        #新的groun truth与先验框有交集且最大，则计算新的坐标，反之保留原来的坐标
        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin  #选择与GT bbox IOU最大的框作为GT bbox，然后循环
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        # Check no annotation label: ignore these anchors...  #对没有标注标签的锚点框做忽视，应该是背景
        # interscts = intersection_with_anchors(bbox) 
        # mask = tf.logical_and(interscts > ignore_threshold,
        #                       label == no_annotation_label)
        # # Replace scores by -1.
        # feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

        return [i+1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]
    # Main loop definition.
    i = 0
    [i, feat_labels, feat_scores,feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                           [i, feat_labels, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])
    # Transform to center / size.   #转换为中心及长宽形式（计算补偿后的中心） 
    feat_cy = (feat_ymax + feat_ymin) / 2. #真实预测值其实是边界框相对于先验框的转换值，encode就是为了求这个转换值 
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features.
    feat_cy = (feat_cy - yref) / href / prior_scaling[0] #(预测真实边界框中心y-参考框中心y)/参考框高/缩放尺度 
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]   #log(预测真实边界框高h/参考框高h)/缩放尺度 
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1) #返回（cx转换值,cy转换值,w转换值,h转换值）形式的边界框的预测值（其实是预测框相对于参考框的转换）  
    #feat_labels特征图上先验框的类别标签(38,38,4)
    #feat_localizations特征图上先验框的位置[(38,38,4),4]，这里如果特征点对应的先验框被认为是正样本时，该先验框对应的坐标使用对应的gt坐标
    #feat_scores特征图上先验框的类别置信度(38,38,4)
    return feat_labels, feat_localizations, feat_scores 
    '''
        经过我们回归得到的变换，经过变换得到真实框，所以这个地方损失函数其实是我们预测的是变换，我们实际的框和anchor之间的变换和我们预测的变换之间的loss。
    我们回归的是一种变换。并不是直接预测框，这个和YOLO是不一样的。和Faster RCNN是一样的。
    '''
def tf_ssd_bboxes_encode(labels, #1D的tensor 包含gt标签
                         bboxes,   #Nx4的tensor包含真实框的相对坐标 
                         anchors,  #参考锚点框信息（y,x,h,w) 其中y,x是中心坐标
                         num_classes,
                         no_annotation_label,
                         ignore_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='ssd_bboxes_encode'):
    """Encode groundtruth labels and bounding boxes using SSD net anchors.
    Encoding boxes for all feature layers.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors: List of Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:  #返回：目标标签，目标位置，目标得分值（都是list形式）
      (target_labels, target_localizations, target_scores):
        Each element is a list of target Tensors.
    """
    with tf.name_scope(scope): 
        target_labels = []        #目标标签
        target_localizations = [] #目标位置
        target_scores = []        #目标得分
        for i, anchors_layer in enumerate(anchors):   #对所有特征图中的参考框做遍历[(y, x, h, w), 6]
            with tf.name_scope('bboxes_encode_block_%i' % i):
                t_labels, t_loc, t_scores = \
                    tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer, #输入真实标签shape(m,)，gt位置大小shape(m,4)，参考框位置大小shape(y, x, h, w) 
                                               num_classes, no_annotation_label,
                                               ignore_threshold,
                                               prior_scaling, dtype)
                target_labels.append(t_labels)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)
        #[(W,H,anchorNum), 6],[(W,H,anchorNum,4), 6],[(W,H,anchorNum), 6]
        return target_labels, target_localizations, target_scores  


def tf_ssd_bboxes_decode_layer(feat_localizations,#解码，在预测时用到，根据之前得到的预测值相对于参考框的转换值后，反推出真实位置（该位置包括真实的x,y,w,h）  
                               anchors_layer,   #需要输入：预测框和参考框的转换feat_localizations，参考框位置尺度信息anchors_layer，以及转换时用到的缩放  
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]): #输出真实预测框的ymin,xmin，ymax,xmax 
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      Tensor Nx4: ymin, xmin, ymax, xmax
    """
    yref, xref, href, wref = anchors_layer #锚点框的参考中心点以及长宽 

    # Compute center, height and width
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes  #预测真实框的坐标信息（两点式的框）


#从预测值 l 中得到边界框的真实位置 b 
def tf_ssd_bboxes_decode(feat_localizations,
                         anchors,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         scope='ssd_bboxes_decode'):
    """Compute the relative bounding boxes from the SSD net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx4: ymin, xmin, ymax, xmax
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_ssd_bboxes_decode_layer(feat_localizations[i],
                                           anchors_layer,
                                           prior_scaling))
        return bboxes


# =========================================================================== #
# SSD boxes selection.
# =========================================================================== #
def tf_ssd_bboxes_select_layer(predictions_layer, localizations_layer, #输入预测得到的类别和位置做筛选
                               select_threshold=None,
                               num_classes=21,
                               ignore_class=0,
                               scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_layer: A SSD prediction layer;
      localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'ssd_bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = tfe.get_shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer,
                                       tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = tfe.get_shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer,
                                         tf.stack([l_shape[0], -1, l_shape[-1]]))

        d_scores = {}
        d_bboxes = {}
        for c in range(0, num_classes):
            if c != ignore_class:  #如果不是背景类别 
                # Remove boxes under the threshold. #去掉低于阈值的box
                scores = predictions_layer[:, :, c] #预测为第c类别的得分值
                fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
                scores = scores * fmask  #保留得分值大于阈值的得分 
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

        return d_scores, d_bboxes #返回字典，每个字典里是对应某类的预测权重和框位置信息；


def tf_ssd_bboxes_select(predictions_net, localizations_net, #输入：SSD网络输出的预测层list；定位层list；类别选择框阈值（None表示都选）  
                         select_threshold=None,   #返回一个字典，key为类别，值为得分和bbox坐标
                         num_classes=21,  #包含了背景类别
                         ignore_class=0,  #第0类是背景
                         scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_net: List of SSD prediction layers;
      localizations_net: List of localization layers;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return: #返回一个字典，其中key是对应类别，值对应得分值和坐标信息 
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, localizations_net]):
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            scores, bboxes = tf_ssd_bboxes_select_layer(predictions_net[i],
                                                        localizations_net[i],
                                                        select_threshold,
                                                        num_classes,
                                                        ignore_class)
            l_scores.append(scores)  #对应某个类别的得分
            l_bboxes.append(bboxes)  #对应某个类别的box坐标信息
        # Concat results.
        d_scores = {}
        d_bboxes = {}
        for c in l_scores[0].keys():
            ls = [s[c] for s in l_scores]
            lb = [b[c] for b in l_bboxes]
            d_scores[c] = tf.concat(ls, axis=1)
            d_bboxes[c] = tf.concat(lb, axis=1)
        return d_scores, d_bboxes


def tf_ssd_bboxes_select_layer_all_classes(predictions_layer, localizations_layer,
                                           select_threshold=None):
    """Extract classes, scores and bounding boxes from features in one layer.
     Batch-compatible: inputs are supposed to have batch-type shapes.

     Args:
       predictions_layer: A SSD prediction layer;
       localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. If None,
        select boxes whose classification score is higher than 'no class'.
     Return: #输出：类别，得分，框
      classes, scores, bboxes: Input Tensors.
     """
    # Reshape features: Batches x N x N_labels | 4
    p_shape = tfe.get_shape(predictions_layer)
    predictions_layer = tf.reshape(predictions_layer,
                                   tf.stack([p_shape[0], -1, p_shape[-1]]))
    l_shape = tfe.get_shape(localizations_layer)
    localizations_layer = tf.reshape(localizations_layer,
                                     tf.stack([l_shape[0], -1, l_shape[-1]]))
    # Boxes selection: use threshold or score > no-label criteria.
    if select_threshold is None or select_threshold == 0:
        # Class prediction and scores: assign 0. to 0-class
        classes = tf.argmax(predictions_layer, axis=2)
        scores = tf.reduce_max(predictions_layer, axis=2)
        scores = scores * tf.cast(classes > 0, scores.dtype)
    else:
        sub_predictions = predictions_layer[:, :, 1:]
        classes = tf.argmax(sub_predictions, axis=2) + 1
        scores = tf.reduce_max(sub_predictions, axis=2)
        # Only keep predictions higher than threshold.
        mask = tf.greater(scores, select_threshold)
        classes = classes * tf.cast(mask, classes.dtype)
        scores = scores * tf.cast(mask, scores.dtype)
    # Assume localization layer already decoded.
    bboxes = localizations_layer
    return classes, scores, bboxes


def tf_ssd_bboxes_select_all_classes(predictions_net, localizations_net,
                                     select_threshold=None,
                                     scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_net: List of SSD prediction layers;
      localizations_net: List of localization layers;
      select_threshold: Classification threshold for selecting a box. If None,
        select boxes whose classification score is higher than 'no class'.
    Return:
      classes, scores, bboxes: Tensors.
    """
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, localizations_net]):
        l_classes = []
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            classes, scores, bboxes = \
                tf_ssd_bboxes_select_layer_all_classes(predictions_net[i],
                                                       localizations_net[i],
                                                       select_threshold)
            l_classes.append(classes)
            l_scores.append(scores)
            l_bboxes.append(bboxes)

        classes = tf.concat(l_classes, axis=1)
        scores = tf.concat(l_scores, axis=1)
        bboxes = tf.concat(l_bboxes, axis=1)
        return classes, scores, bboxes #返回所有特征图综合得出的类别，得分

