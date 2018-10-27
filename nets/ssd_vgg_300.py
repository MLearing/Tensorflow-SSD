# Copyright 2016 Paul Balanca. All Rights Reserved.
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
"""Definition of 300 VGG-based SSD network.

This model was initially introduced in:
SSD: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
https://arxiv.org/abs/1512.02325

Two variants of the model are defined: the 300x300 and 512x512 models, the
latter obtaining a slightly better accuracy on Pascal VOC.

Usage:
    with slim.arg_scope(ssd_vgg.ssd_vgg()):
        outputs, end_points = ssd_vgg.ssd_vgg(inputs)

This network port of the original Caffe model. The padding in TF and Caffe
is slightly different, and can lead to severe accuracy drop if not taken care
in a correct way!

In Caffe, the output size of convolution and pooling layers are computing as
following: h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1

Nevertheless, there is a subtle difference between both for stride > 1. In
the case of convolution:
    top_size = floor((bottom_size + 2*pad - kernel_size) / stride) + 1
whereas for pooling:
    top_size = ceil((bottom_size + 2*pad - kernel_size) / stride) + 1
Hence implicitely allowing some additional padding even if pad = 0. This
behaviour explains why pooling with stride and kernel of size 2 are behaving
the same way in TensorFlow and Caffe.

Nevertheless, this is not the case anymore for other kernel sizes, hence
motivating the use of special padding layer for controlling these side-effects.

@@ssd_vgg_300
"""
import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import ssd_common

slim = tf.contrib.slim


# =========================================================================== #
# SSD class definition.
# =========================================================================== #
#collectionsģ���namedtuple���಻������ʹ��item��index����item��������ͨ��item��name
#���з��ʿ��Խ�namedtuple����Ϊc�е�struct�ṹ�������Ƚ�����item������Ȼ���ÿ��item��������
SSDParams = namedtuple('SSDParameters', ['img_shape',          #����ͼ���С
                                         'num_classes',        #���������
                                         'no_annotation_label',#�ޱ�ע��ǩ
                                         'feat_layers',        #������
                                         'feat_shapes',        #��������״��С
                                         'anchor_size_bounds', #ê����С���±߽磬����ԭͼ��ȵõ���С��ֵ 
                                         'anchor_sizes',       #��ʼê���ߴ�
                                         'anchor_ratios',      #ê��򳤿��� 
                                         'anchor_steps',       #����ͼ���ԭʼͼ�������  
                                         'anchor_offset',      #ê������ĵ�ƫ�� 
                                         'normalizations',     #�Ƿ����� 
                                         'prior_scaling'       #�Ƕ�����ͼ�ο�����gtbox���ع�ʱ�õ��ĳ߶����ţ�0.1,0.1,0.2,0.2��  
                                         ])

#����ssdNet������󲢳�ʼ������
class SSDNet(object):
    """Implementation of the SSD VGG-based 300 network.

    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.
    """
    #��������
    default_params = SSDParams(
        img_shape=(300, 300),  #����ͼƬ��С
        num_classes=21,    #�����������ڣ���21��Ŀ�����  
        no_annotation_label=21,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'], #��ȡ�����Ĳ�
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)], #��������״��С
        anchor_size_bounds=[0.15, 0.90],
        # anchor_size_bounds=[0.20, 0.90],  #�����г�ʼԤ����СΪ0.2x300~0.9x300��ʵ�ʴ�����[45,270]  
        anchor_sizes=[(21., 45.),#ֱ�Ӹ�����ÿ������ͼ�������ê����С�����һ����������С��h:21;w:45;  ��6������ͼ���ڻع�  
                      (45., 99.),#ԽС�Ŀ��ܹ��õ�ԭͼ�ϸ���ľֲ���Ϣ����֮�õ������ȫ����Ϣ��
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        # anchor_sizes=[(30., 60.),
        #               (60., 111.),
        #               (111., 162.),
        #               (162., 213.),
        #               (213., 264.),
        #               (264., 315.)],
        anchor_ratios=[[2, .5],    #ÿ���������ϵ�ÿ��������Ԥ���box�����ȼ��������磺block4: def_boxes:4 
                       [2, .5, 3, 1./3], #block7: def_boxes:6   ��ratios�е�4��+Ĭ�ϵ�1:1+�������ӵ�һ��=6��{1,1',2,0.5,3,1/3}
                       [2, .5, 3, 1./3], #block8: def_boxes:6 
                       [2, .5, 3, 1./3], #block9: def_boxes:6 
                       [2, .5],  #block10: def_boxes:4 
                       [2, .5]], #block11: def_boxes:4   #��ע��ʵ������ȥ��Ĭ�ϵ�ratio=1�Լ������һ��sqrt(��ʼ���*��ʼ���)�����������  
        anchor_steps=[8, 16, 32, 64, 100, 300], #����ͼê���Ŵ�ԭʼͼ�����ű���������Ұ��
        anchor_offset=0.5,     #ÿ��ê������ĵ��ڸ�����ͼcell���ģ����offset=0.5 
        normalizations=[20, -1, -1, -1, -1, -1], #�Ƿ��һ��������0����У���������һ����Ŀǰ����ֻ��block_4�������򻯣���Ϊ�ò�ȽϿ�ǰ����norm�ϴ�����L2���򻯣�������ÿ��������channelά������һ�����Ա�֤�ͺ��������첻�Ǻܴ�  
        prior_scaling=[0.1, 0.1, 0.2, 0.2] #����ͼ��ÿ��Ŀ����ο����ĳߴ����ţ�y,x,h,w������ʱ�õ� 
        )

   
    def __init__(self, params=None): #��������ĳ�ʼ��
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams): #�Ƿ��в������룬����������ģ�����ʹ��Ĭ�ϵ�
            self.params = params #isinstance��python�ăȽ��������������1�����2��������ͬ�򷵻�true�� 
        else:
            self.params = SSDNet.default_params

    # ======================================================================= #
    def net(self, inputs,              #��������ģ��
            is_training=True,          #�Ƿ�ѵ��
            update_feat_shapes=True,   #�Ƿ����������ĳߴ�
            dropout_keep_prob=0.5,     #dropout=0.5
            prediction_fn=slim.softmax,#����softmaxԤ����
            reuse=None,                #�����ж�
            scope='ssd_300_vgg'):      #��������ssd_300_vgg   ����������ʱVGG������ѵ��ͼ��size��300x300��
        """SSD network definition.
        """
        r = ssd_net(inputs,      # r=[predictions, localisations, logits, end_points]
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!) #�����ⲽ�ҵ���������ö������и�������������룬δ�������н��ܵ��Ǽ���block
        if update_feat_shapes:  #�Ƿ����������ͼ��ߴ磿
            #��������ΪĬ�ϳ�ȡ�����Ĳ��ǣ�4,7,8,9,10,11����Ҳ�����Լ�ѡ���ȡ�Ĳ㣬�����ȡ6�㣬���ݳ�ȡ�����Ĭ�ϲ���
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes) #Ԥ��������ߴ�͸����������ߴ磬������º������ͼ�ߴ��б�  
            self.params = self.params._replace(feat_shapes=shapes)#�����µ�����ͼ�ߴ�shapes�滻��ǰ������ͼ�ߴ�
        return r    #���������������r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'): #����Ȩ��˥��=0.0005��L2������ϵ��������������NHWC
        """Network arg_scope.
        """
        return ssd_arg_scope(weight_decay, data_format=data_format)

    def arg_scope_caffe(self, caffe_scope):
        """Caffe arg_scope used for weights importing.
        """
        return ssd_arg_scope_caffe(caffe_scope)

    # ======================================================================= #
    def update_feature_shapes(self, predictions): #����������״�ߴ磨����Ԥ������
        """Update feature shapes from predictions collection (Tensor or Numpy
        array).
        """
        shapes = ssd_feat_shapes_from_net(predictions, self.params.feat_shapes)
        self.params = self.params._replace(feat_shapes=shapes)

    def anchors(self, img_shape, dtype=np.float32): #����ԭʼͼ��ߴ磻����ÿ��������ÿ���ο�ê����λ�ü��ߴ���Ϣ��x,y,h,w��
        """Compute the default anchor boxes, given an image shape.
        """
        # first->[((38*38*1),(38*38*1),(4*1),(4*1)),6]
        return ssd_anchors_all_layers(img_shape,  #���Ǹ��ؼ���������������������еĲο�ê���λ�úͳߴ���Ϣ
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)
                                      
                                      
    #���룬���ڽ���ǩ��Ϣ����ʵĿ����Ϣ��ê�����Ϣ������һ�𣻵õ�Ԥ����ʵ�򵽲ο����ת��ֵ  
    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,  #δ��ע�ı�ǩ��Ӧ�ô���������
            ignore_threshold=0.5,     #IOUɸѡ��ֵ
            prior_scaling=self.params.prior_scaling, #����ͼĿ����ο����ĳߴ����ţ�0.1,0.1,0.2,0.2��
            scope=scope)

    #���룬��ê�����Ϣ��ê�����Ԥ����ʵ����ת��ֵ���õ����ǵ�Ԥ���ymin,xmin,ymax,xmax��  
    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    #ͨ��SSD���磬�õ���⵽��bbox
    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip #ѡȡtop_k=400���򣬲��Կ����޽�������ԭͼ�ߴ緶Χ���е���
        rscores, rbboxes = \       #�õ���Ӧĳ�����ĵ÷�ֵ�Լ�bbox
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \   #���յ÷ָߵͣ�ɸѡ��400��bbox�Ͷ�Ӧ�÷�
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \  #Ӧ�÷Ǽ���ֵ���ƣ�ɸѡ����÷����bbox�ص��ʴ���0.5�ģ�����200��  
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        if clipping_bbox is not None:
            rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes   #���زü��õ�bbox�Ͷ�Ӧ�÷�
    #����һ��ground?truth�������������ƥ�䣬����ground?truth����������̫���ˣ�??
    #���Ը����������������ܶࡣΪ�˱�֤������������ƽ�⣬SSD������hard?negative?mining��??
    #���ǶԸ��������г���������ʱ�������Ŷ���Ԥ�ⱳ�������Ŷ�ԽС�����Խ�󣩽��н������У�??
    #ѡȡ���Ľϴ��top-k��Ϊѵ���ĸ��������Ա�֤�������������ӽ�1:3??

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)


# =========================================================================== #
# SSD tools...
# =========================================================================== #
def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(300, 300)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (300 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * size_bounds[0] / 2, img_size * size_bounds[0]]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes


#��Ԥ�������ʲ��ȡ���������״ predictions->[N,W,H,num_anchors,classes��6]
def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:  #l:��Ԥ���������״[N,W,H,num_anchors,classes]
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray): #���l��np.ndarray���ͣ���l����״����shape������shape��Ϊlist 
            shape = l.shape  #[N,W,H,num_anchors,classes]
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]   #[W,H,num_anchors] ��������
        # Problem: undetermined shape...
        if None in shape:  #���Ԥ��������ߴ�δ������ʹ��Ĭ�ϵ���״������shape�е�ֵ����������״�б���
            return default_shapes #default_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        else:
            feat_shapes.append(shape)
    return feat_shapes    #���ظ��º�������ߴ�list


#��ⵥ������ͼ������ê�������ͳߴ���Ϣ<���������ͼ>
def ssd_anchor_one_layer(img_shape, ,#ԭʼͼ��shape,(300*300) 
                         feat_shape, #����ͼshape   ,first->(38,38)
                         sizes,  #Ԥ���box size,first->(21., 45.)
                         ratios, #aspect ����,first->(2, .5)
                         step,   #anchor�Ĳ�,first->(8)
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.              #�������ĵ�Ĺ�һ������
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values... #��һ����ԭͼ��ê���������꣨x,y��;������ֵ��Ϊ(0,1) 
    """    
    #�����У���������    
    feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]    
    anchor_sizes=[(21., 45.),                      
                  (45., 99.),                      
                  (99., 153.),                      
                  (153., 207.),                      
                  (207., 261.),                      
                  (261., 315.)]    
    anchor_ratios=[ [2, .5],                       
                    [2, .5, 3, 1./3],                       
                    [2, .5, 3, 1./3],                       
                    [2, .5, 3, 1./3],                       
                    [2, .5],                       
                    [2, .5]]    
    anchor_steps=[8, 16, 32, 64, 100, 300]      
    offset=0.5     
    dtype=np.float32     
    feat_shape=feat_shapes[0]    
    step=anchor_steps[0]    
    """ 
    #-------------------------------------------|
    #�����У�y��x��shapeΪ��38,38����38,38��               |   
    #-------------------------------------------|
    #y��ʾ��ͼ���H���������µ�����ͬһ��H��ͬ                     |
    #y��ֵΪ                                       |
    #array([[ 0,  0,  0, ...,  0,  0,  0],      |
    #       [ 1,  1,  1, ...,  1,  1,  1],      |
    #       [ 2,  2,  2, ...,  2,  2,  2],      |
    #         ...,                              |
    #       [35, 35, 35, ..., 35, 35, 35],      |
    #       [36, 36, 36, ..., 36, 36, 36],      |
    #       [37, 37, 37, ..., 37, 37, 37]])     |
    #-------------------------------------------|
    #x��ʾ��ͼ���W���������ҵ�����ͬһ��W��ͬ                     |
    #x��ֵΪ                                       |
    #array([[ 0,  1,  2, ...,  35,  36,  37],   |  
    #       [ 0,  1,  2, ...,  35,  36,  37],   | 
    #       [ 0,  1,  2, ...,  35,  36,  37],   | 
    #         ...,                              |
    #       [ 0,  1,  2, ...,  35,  36,  37],   | 
    #       [ 0,  1,  2, ...,  35,  36,  37],   |  
    #       [ 0,  1,  2, ...,  35,  36,  37]])  |
    #-------------------------------------------

    #���ڵ�һ������ͼ��block4��38x38����y=[[0,0,����0],[1,1,����1]������[37,37,������37]]����x=[[0,1,2������37]��[0,1,2������37],����[0,1,2������37]]  
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]  
    #������y=(y+0.5)��8/300,x=(x+0.5)��8/300
    #��38��cell��Ӧê����y����ƫ����ÿ��cell���ģ�Ȼ����Ը���Ұ���ٳ���ԭͼ  
    #���Եõ���ԭͼ�ϣ����ԭͼ������С��ÿ��ê����������x,y 
    #�Ե�һ��Ԫ��Ϊ�����򵥷���Ϊ����0+0.5)/38��Ϊ��Ծ��룬��SSD-Caffeʹ�õ��ǣ�0+0.5��*step/img_shape
    y = (y.astype(dtype) + offset) * step / img_shape[0]  #astypeת����������
    x = (x.astype(dtype) + offset) * step / img_shape[1]
    # Expand dims to support easy broadcasting. #��չά�ȣ�ά�ȱ�Ϊ��38,38,1����ԭά��Ϊ��38,38��
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)#������ͼ��ÿ�����Ӧ��ê�������;�磺���ڵ�һ������ͼÿ����Ԥ��4��ê���block4��38x38����2+2=4  
    h = np.zeros((num_anchors, ), dtype=dtype) #���ڵ�һ������ͼ��h��shape=4x��w��shape=4x 
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.����һ����
    h[0] = sizes[0] / img_shape[0]  #��һ��ê���ĸ�h[0]=��ʼê��ĸ�/ԭͼ��С�ĸߣ����磺h[0]=21/300 
    w[0] = sizes[0] / img_shape[1]  #��һ��ê���Ŀ�w[0]=��ʼê��Ŀ�/ԭͼ��С�Ŀ������磺h[0]=21/300
    di = 1 #ê�������ƫ�� 
    if len(sizes) > 1: 
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]  #�ڶ���ê���ĸ�h[1]=sqrt����ʼê��ĸ�*��ʼê��Ŀ���/ԭͼ��С�ĸߣ����磺h[1]=sqrt(21*45)/300  
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]  #�ڶ���ê���ĸ�w[1]=sqrt����ʼê��ĸ�*��ʼê��Ŀ���/ԭͼ��С�Ŀ������磺w[1]=sqrt(21*45)/300  
        di += 1
    for i, r in enumerate(ratios): #����������������һ������ͼ��rֻ��������2��0.5�����ĸ�ê���size��h[0]~h[3]��  
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r) #���磺���ڵ�һ������ͼ��h[0+2]=h[2]=21/300/sqrt(2);w[0+2]=w[2]=45/300*sqrt(2)  
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r) #���磺���ڵ�һ������ͼ��h[1+2]=h[3]=21/300/sqrt(0.5);w[1+2]=w[3]=45/300*sqrt(0.5)   
    return y, x, h, w   #���ع�һ����ê������ͳߴ� 

#�����������ͼ������ê�������ͳߴ���Ϣ
def ssd_anchors_all_layers(img_shape,  #�����������ͼ��ê�����ĸ�������Ϣ�� ����ԭʼͼ��С[300*300]
                           layers_shape,   #ÿ����������״�ߴ�[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
                           anchor_sizes,   #��ʼ����ͼ�п�ĳ���size anchor_sizes=[(21., 45.),(45., 99.),(99., 153.),(153., 207.),(207., 261.),(261., 315.)],
                           anchor_ratios,  #ê��򳤿����б�[[2, .5],    [2, .5, 3, 1./3], [2, .5, 3, 1./3],  [2, .5, 3, 1./3], [2, .5],  [2, .5]]
                           anchor_steps,   #ê������ԭͼ���ű���[8, 16, 32, 64, 100, 300]
                           offset=0.5,     #ê��������ÿ������ͼcell�е�ƫ��
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = [] #���ڴ����������ͼ��ê���λ�óߴ���Ϣ 
    for i, s in enumerate(layers_shape):  #6������ͼ�ߴ磻�磺��0����38x38
        # first->anchor_bboxes=[y, x, h, w]=[(38*38*1),(38*38*1),(4*1),(4*1)]
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,    #�ֱ����ÿ������ͼ��ê����λ�óߴ���Ϣ��
                                             anchor_sizes[i], #���룺��i������ͼ����ʼê����С�����0����(21., 45.) 
                                             anchor_ratios[i],#���룺��i������ͼ��ê��򳤿����б������0����[2, .5]
                                             anchor_steps[i], #���룺��i������ͼ��ê������ԭʼͼ�����űȣ����0����8 
                                             offset=offset, dtype=dtype)#���룺ê��������ÿ������ͼcell�е�ƫ��
        layers_anchors.append(anchor_bboxes) #��6������ͼ��ÿ������ͼ�ϵĵ��Ӧ��ê���6����4��������
    return layers_anchors #[[(38*38*1),(38*38*1),(4*1),(4*1)], 6]


# =========================================================================== #
//**+*# Functional definition of VGG-based SSD 300.
# =========================================================================== #
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():

         return x.get_shape().as_list()
    else:
        # get_shape����ֵ��with_rank�൱�ڶ���assert���Ƿ�rankΪָ��ֵ
        static_shape = x.get_shape().with_rank(rank).as_list()  #�ж���״�ǲ���rankά��
        # tf.shape��������������num����Ϊ"The length of the dimension `axis`."��axisĬ��Ϊ0
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        # list���ж���ĸ����֣�û�еĸ�tensor
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

#ÿ���������λ��ƫ�ƺ�������Ŷ�
def ssd_multibox_layer(inputs,  #����������
                       num_classes,   #�����    21
                       sizes,         #�ο������ĳ߶� first->(21,45)
                       ratios=[1],    #Ĭ�ϵ�����򳤿���Ϊ first->[2,0.5,3,1/3]
                       normalization=-1, #Ĭ�ϲ�������
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0: #������������������L2����
        net = custom_layers.l2_normalization(net, scaling=True) #��ͨ������ά�Ƚ������򻯣�������gamma����ϵ��
    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)#ÿ������ͼ�ο������ĸ���[4,6,6,6,4,4]

    # Location. #ÿ��������Ӧ4��������Ϣ
    num_loc_pred = num_anchors * 4   #����ͼ��ÿ����ԪԤ�����������ά��=ê�����*4 
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,#ͨ��������ͼ����3x3�����õ�λ����Ϣ
                           scope='conv_loc') #�ò����Ƕ�λ��Ϣ�����ά��Ϊ[N,����ͼW,����ͼH,ÿ����Ԫ����ê�������num_anchors * 4]  
    loc_pred = custom_layers.channel_to_last(loc_pred) # ensure data format be "NWHC"
    loc_pred = tf.reshape(loc_pred,  #�����������ͼ����ê���Ԥ��Ŀ��λ��,tensorΪ[N��W��H, num_anchors��4]  
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.  #���Ԥ��  
    num_cls_pred = num_anchors * num_classes #����ͼ��ÿ����ԪԤ����������ά��=ê�����*������  
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls') #�ò����������Ϣ�����ά��Ϊ[N,����ͼW,����ͼH,ÿ����Ԫ����ê����Ӧ�����Ϣ]  
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,  #���õ���feature maps reshapeΪ[N��W��H, num_anchors��������]  
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    return cls_pred, loc_pred  #����Ԥ��õ�������boxλ�� 


#����ssd����ṹ
def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,      #������
            feat_layers=SSDNet.default_params.feat_layers,      #������ 
            anchor_sizes=SSDNet.default_params.anchor_sizes,    
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,#����
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
    """SSD net definition.
    """
    # if data_format == 'NCHW':
    #     inputs = tf.transpose(inputs, perm=(0, 3, 1, 2))

    # End_points collect relevant activations for external use.
    end_points = {} #�����ռ�ÿһ��������
    with tf.variable_scope(scope, 'ssd_300_vgg', [inputs], reuse=reuse): #tf.variable_scope()����ָ��������������reuse���ñ��������ã��ο�https://blog.csdn.net/wanglitao588/article/details/76976428
        # Original VGG-16 blocks.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1') #VGG16����ĵ�һ��conv���ظ�2�ξ�������Ϊ3x3,64������  
        end_points['block1'] = net     #conv1_2�������end_points��name='block1'  
        net = slim.max_pool2d(net, [2, 2], scope='pool1')  #�ػ���
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')  #�ظ�2�ξ�������Ϊ3x3,128������  
        end_points['block2'] = net   #conv2_2�������end_points��name='block2'  
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')  #�ظ�3�ξ�������Ϊ3x3,256������ 
        end_points['block3'] = net   #conv3_3�������end_points��name='block3'  
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')  #�ظ�3�ξ�������Ϊ3x3,512������,���[batch,38,38,512]
        end_points['block4'] = net   #conv4_3�������end_points��name='block4'  
        net = slim.max_pool2d(net, [2, 2], scope='pool4') 
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')   #�ظ�3�ξ�������Ϊ3x3,512������
        end_points['block5'] = net   #conv5_3�������end_points��name='block5'  
        net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

        # Additional SSD blocks. #��ӵ�SSD��
        # Block 6: let's dilate the hell out of it! #ȥ����VGG��ȫ���Ӳ� 
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6') #��VGG�����������ĳػ���������չ���������׾�������  
        end_points['block6'] = net   #conv6�������end_points��name='block6'  
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training) #dropout�� 
        # Block 7: 1x1 conv. Because the fuck.
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7') #��dropout���������1x1���������1024������name='block7'�����[batch,19,19,1024] 
        end_points['block7'] = net  
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training) #������������������dropout

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'  #������dropout��������1x1������Ȼ����3x3������,���512����ͼ��name=��block8��,���[batch,10,10,512]  
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID') 
        end_points[end_point] = net
        end_point = 'block9'  #������������1x1������Ȼ����3x3���������256����ͼ��name=��block9��,���[batch,5,5,256] 
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block10' #������������1x1������Ȼ����3x3���������256����ͼ��name=��block10��,���[batch,3,3,256]   
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block11' #������������1x1������Ȼ����3x3���������256����ͼ��name=��block11�������[batch,1,1,256]   
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net

        # Prediction and localisations layers. #Ԥ��Ͷ�λ 
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):  #����������
            with tf.variable_scope(layer + '_box'): #���������Χ
                # p->[N��W��H, num_anchors��������,6��] 
                p, l = ssd_multibox_layer(end_points[layer],#����߶ȴ�СboxԤ��������㣬����ÿ��cell��ÿ�������Ԥ������p��Ԥ���λ��l  
                                          num_classes,      #������
                                          anchor_sizes[i],  #�����߶ȣ�ͬһ����ͼ�ϵ������߶Ⱥͳ�����һ�£� 
                                          anchor_ratios[i], #����򳤿���
                                          normalizations[i])#ÿ������������Ϣ��Ŀǰ��ֻ�Ե�һ������ͼ����һ������
            #��ÿһ���Ԥ���ռ�
            predictions.append(prediction_fn(p))#prediction_fnΪsoftmax��Ԥ�������� ����ʧ�����е�-c_i^p��
            logits.append(p)  #��ÿ��cellÿ�������Ԥ������ĸ���ֵ����logits�� (��ʧ�����е�c_i^p)
            localisations.append(l) #Ԥ��λ����Ϣ

        #����ĳ�����ĸ��ʣ�λ��Ԥ�������������Ԥ�������Ŷȣ��Լ�������  
        return predictions, localisations, logits, end_points
ssd_net.default_image_size = 300

#����Լ�������еĳ������趨������Ĭ�ϲ����������ǲ����ú����Ĳ���ʱ��Ĭ�������������
#�ο���https://blog.csdn.net/u013921430/article/details/80915696
#      https://blog.csdn.net/DeepOscar/article/details/82762929?utm_source=blogxgwz3
def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'): #Ȩ��˥��ϵ��=0.0005������L2�������ϵ��
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    # ��������������Ǹ�list_ops�е���������Ĭ��ֵ������ÿ��list_ops�е�ÿ����Ա��Ҫ��@add_arg_scope���β��С�
    # ���������еķ���������arg_scope����Ĭ�ϲ���, ֻ����@slim.add_arg_scope���ι��ķ�������ʹ��arg_scope
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc


# =========================================================================== #
# Caffe scope: importing weights at initialization.
# =========================================================================== #
def ssd_arg_scope_caffe(caffe_scope):
    """Caffe scope definition.

    Args:
      caffe_scope: Caffe scope object with loaded weights.

    Returns:
      An arg_scope.
    """
    # Default network arg scope.
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=caffe_scope.conv_weights_init(),
                        biases_initializer=caffe_scope.conv_biases_init()):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([custom_layers.l2_normalization],
                                scale_initializer=caffe_scope.l2_norm_scale_init()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    return sc


# =========================================================================== #
# SSD loss function.
# =========================================================================== #
#��ʧ��������Ϊλ���������Ŷ����ļ�Ȩ�ͣ�����ĳһlayerԤ����������״ΪS1->[N,H,W,anchorNum,21]
def ssd_losses(logits,        #Ԥ��������[S1,S2,S3,S4,S5,S6]
               localisations, #Ԥ��λ��[(N,H,W,anchorNum,4),6��)]
               gclasses,      #ground truth �����������[(N,W,H,anchorNum), 6]
               glocalisations,#ground truth ��������λ��ƫ�� [(N,W,H,anchorNum,4), 6]
               gscores,       #ground truth �������Ľ����ȵ÷֣����Ŷȣ�[(N,W,H,anchorNum), 6]
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,     #λ�����Ȩ��ϵ��
               label_smoothing=0.,
               device='/cpu:0',
               scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        lshape = tfe.get_shape(logits[0], 5) # logits[0] = S1->[N,H,W,anchorNum,21]
        num_classes = lshape[-1]  # 21
        batch_size = lshape[0]    # N

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)): # len(logits) = 6��feature map
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))  #��Ԥ�����ĸ���ֵreshape�ɣ�N*H*W*anchorNum,21��
            fgclasses.append(tf.reshape(gclasses[i], [-1]))           #��ʵ���N*H*W*anchorNum,��
            fgscores.append(tf.reshape(gscores[i], [-1]))             #��ʵĿ��ĵ÷֣�N*H*W*anchorNum,��
            flocalisations.append(tf.reshape(localisations[i], [-1, 4])) #Ԥ��Ŀ��߿����� ��N*H*W*anchorNum,4��
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4])) #���ڽ���ʵĿ��gt��������б���洢 
        # And concat the crap! ƴ�ӣ�����ԭ����[[S1],[S2],[S3],[S4],[S5],[S6]]->[S1,S2,S3,S4,S5,S6]
        logits = tf.concat(flogits, axis=0)   # Ԥ�����[(N*H*W*anchorNum,21),6]
        gclasses = tf.concat(fgclasses, axis=0) #�������� [(N*H*W*anchorNum,),6]
        gscores = tf.concat(fgscores, axis=0)   #���������ʵ��IOU(���Ŷ�) [(N*H*W*anchorNum,),6]
        localisations = tf.concat(flocalisations, axis=0)   # Ԥ��λ��[(N*H*W*anchorNum,4),6]
        glocalisations = tf.concat(fglocalisations, axis=0) # �����λ��[(N*H*W*anchorNum,4),6]
        dtype = logits.dtype

        # Matching strategy,Compute positive matching mask...
        pmask = gscores > match_threshold #Ԥ�������ʵ��IOU>0.5�����������Ϊ������
        fpmask = tf.cast(pmask, dtype)  #��������־
        n_positives = tf.reduce_sum(fpmask) #��������Ŀ

        # Hard negative mining...
        #Ϊ�˱�֤������������ƽ�⣬SSD������hard negative mining�����ǶԸ��������г�����
        #����ʱ�������Ŷ���Ԥ�ⱳ�������Ŷ�ԽС�����Խ�󣩽��н������У�
        #ѡȡ���Ľϴ��top-k��Ϊѵ���ĸ��������Ա�֤�������������ӽ�1:3  
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)   #���Ԥ��[(N*H*W*anchorNum,21),6]
        nmask = tf.logical_and(tf.logical_not(pmask),gscores > -0.5) #������
        fnmask = tf.cast(nmask, dtype) #��������־
        nvalues = tf.where(nmask,  #���������Ŷ� [(N*H*W*anchorNum,),6]
                           predictions[:, 0], #���������Ŷȣ���21������е�0��������
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1]) #���һά(N*H*W*anchorNum*6,)
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32) #����������Ŀ
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size#��������������֤��������3�� 
        n_neg = tf.minimum(n_neg, max_neg_entries)  #��֤ѡȡ�ĸ����������������ܵĸ���������

        #����ע����-nvalues_flat���õ���valҲ�Ǹ����������������У��൱����������
        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)  #����ʱ�������Ŷ���Ԥ�ⱳ�������Ŷ�ԽС�����Խ�󣩽��н������У�ѡȡ���Ľϴ��top-k��Ϊѵ���ĸ�����  
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype) #��������־

        # Add cross-entropy loss.  #�����أ�����L_conf��
        with tf.name_scope('cross_entropy_pos'): #������
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,   #Ԥ��������Ŷ�logits��c_i^p��->[(N*H*W*anchorNum,21),6]
                                                                  labels=gclasses) #gclasses->[(N*H*W*anchorNum,),6]
            loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value') #�����Ŷ��������������������batch-size  
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_neg'): #������
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)
            loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        # Add localization loss: smooth L1, L2, ... #(����L_loc)
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localisations - glocalisations) #Ԥ��λ��ƫ��-��ʵλ��ƫ��ֵ��Ȼ����Smooth L1 loss  
            loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')#�������loss*Ȩ�أ�=alpha*������������ͺ����batch-size  
            tf.losses.add_loss(loss) #������Ŷ�����λ�����ļ�Ȩ��


def ssd_losses_old(logits, localisations,
                   gclasses, glocalisations, gscores,
                   match_threshold=0.5,
                   negative_ratio=3.,
                   alpha=1.,
                   label_smoothing=0.,
                   device='/cpu:0',
                   scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors;
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
    """
    with tf.device(device):
        with tf.name_scope(scope, 'ssd_losses'):
            l_cross_pos = []
            l_cross_neg = []
            l_loc = []
            for i in range(len(logits)):
                dtype = logits[i].dtype
                with tf.name_scope('block_%i' % i):
                    # Sizing weight...
                    wsize = tfe.get_shape(logits[i], rank=5)
                    wsize = wsize[1] * wsize[2] * wsize[3]

                    # Positive mask.
                    pmask = gscores[i] > match_threshold
                    fpmask = tf.cast(pmask, dtype)
                    n_positives = tf.reduce_sum(fpmask)

                    # Select some random negative entries.
                    # n_entries = np.prod(gclasses[i].get_shape().as_list())
                    # r_positive = n_positives / n_entries
                    # r_negative = negative_ratio * n_positives / (n_entries - n_positives)

                    # Negative mask.
                    no_classes = tf.cast(pmask, tf.int32)
                    predictions = slim.softmax(logits[i])
                    nmask = tf.logical_and(tf.logical_not(pmask),
                                           gscores[i] > -0.5)
                    fnmask = tf.cast(nmask, dtype)
                    nvalues = tf.where(nmask,
                                       predictions[:, :, :, :, 0],
                                       1. - fnmask)
                    nvalues_flat = tf.reshape(nvalues, [-1])
                    # Number of negative entries to select.
                    n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                    n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)
                    n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)
                    max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
                    n_neg = tf.minimum(n_neg, max_neg_entries)

                    val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                    max_hard_pred = -val[-1]
                    # Final negative mask.
                    nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
                    fnmask = tf.cast(nmask, dtype)

                    # Add cross-entropy loss.
                    with tf.name_scope('cross_entropy_pos'):
                        fpmask = wsize * fpmask
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                              labels=gclasses[i])
                        loss = tf.losses.compute_weighted_loss(loss, fpmask)
                        l_cross_pos.append(loss)

                    with tf.name_scope('cross_entropy_neg'):
                        fnmask = wsize * fnmask
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                              labels=no_classes)
                        loss = tf.losses.compute_weighted_loss(loss, fnmask)
                        l_cross_neg.append(loss)

                    # Add localization loss: smooth L1, L2, ...
                    with tf.name_scope('localization'):
                        # Weights Tensor: positive mask + random negative.
                        weights = tf.expand_dims(alpha * fpmask, axis=-1)
                        loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
                        loss = tf.losses.compute_weighted_loss(loss, weights)
                        l_loc.append(loss)

            # Additional total losses...
            with tf.name_scope('total'):
                total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
                total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
                total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
                total_loc = tf.add_n(l_loc, 'localization')

                # Add to EXTRA LOSSES TF.collection
                tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
                tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
                tf.add_to_collection('EXTRA_LOSSES', total_cross)
                tf.add_to_collection('EXTRA_LOSSES', total_loc)