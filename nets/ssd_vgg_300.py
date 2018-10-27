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
#collections模块的namedtuple子类不仅可以使用item的index访问item，还可以通过item的name
#进行访问可以将namedtuple理解为c中的struct结构，其首先将各个item命名，然后对每个item赋予数据
SSDParams = namedtuple('SSDParameters', ['img_shape',          #输入图像大小
                                         'num_classes',        #分类类别数
                                         'no_annotation_label',#无标注标签
                                         'feat_layers',        #特征层
                                         'feat_shapes',        #特征层形状大小
                                         'anchor_size_bounds', #锚点框大小上下边界，是与原图相比得到的小数值 
                                         'anchor_sizes',       #初始锚点框尺寸
                                         'anchor_ratios',      #锚点框长宽比 
                                         'anchor_steps',       #特征图相对原始图像的缩放  
                                         'anchor_offset',      #锚点框中心的偏移 
                                         'normalizations',     #是否正则化 
                                         'prior_scaling'       #是对特征图参考框向gtbox做回归时用到的尺度缩放（0.1,0.1,0.2,0.2）  
                                         ])

#创建ssdNet网络对象并初始化参数
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
    #参数设置
    default_params = SSDParams(
        img_shape=(300, 300),  #输入图片大小
        num_classes=21,    #包含背景在内，共21类目标类别  
        no_annotation_label=21,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'], #抽取特征的层
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)], #特征层形状大小
        anchor_size_bounds=[0.15, 0.90],
        # anchor_size_bounds=[0.20, 0.90],  #论文中初始预测框大小为0.2x300~0.9x300；实际代码是[45,270]  
        anchor_sizes=[(21., 45.),#直接给出的每个特征图上起初的锚点框大小；如第一个特征层框大小是h:21;w:45;  共6个特征图用于回归  
                      (45., 99.),#越小的框能够得到原图上更多的局部信息，反之得到更多的全局信息；
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
        anchor_ratios=[[2, .5],    #每个特征层上的每个特征点预测的box长宽比及数量；如：block4: def_boxes:4 
                       [2, .5, 3, 1./3], #block7: def_boxes:6   （ratios中的4个+默认的1:1+额外增加的一个=6）{1,1',2,0.5,3,1/3}
                       [2, .5, 3, 1./3], #block8: def_boxes:6 
                       [2, .5, 3, 1./3], #block9: def_boxes:6 
                       [2, .5],  #block10: def_boxes:4 
                       [2, .5]], #block11: def_boxes:4   #备注：实际上略去了默认的ratio=1以及多加了一个sqrt(初始框宽*初始框高)，后面代码有  
        anchor_steps=[8, 16, 32, 64, 100, 300], #特征图锚点框放大到原始图的缩放比例（感受野）
        anchor_offset=0.5,     #每个锚点框中心点在该特征图cell中心，因此offset=0.5 
        normalizations=[20, -1, -1, -1, -1, -1], #是否归一化，大于0则进行，否则不做归一化；目前看来只对block_4进行正则化，因为该层比较靠前，其norm较大，需做L2正则化（仅仅对每个像素在channel维度做归一化）以保证和后面检测层差异不是很大；  
        prior_scaling=[0.1, 0.1, 0.2, 0.2] #特征图上每个目标与参考框间的尺寸缩放（y,x,h,w）解码时用到 
        )

   
    def __init__(self, params=None): #网络参数的初始化
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams): #是否有参数输入，是则用输入的，否则使用默认的
            self.params = params #isinstance是python的內建函数，如果参数1与参数2的类型相同则返回true； 
        else:
            self.params = SSDNet.default_params

    # ======================================================================= #
    def net(self, inputs,              #定义网络模型
            is_training=True,          #是否训练
            update_feat_shapes=True,   #是否更新特征层的尺寸
            dropout_keep_prob=0.5,     #dropout=0.5
            prediction_fn=slim.softmax,#采用softmax预测结果
            reuse=None,                #网络判断
            scope='ssd_300_vgg'):      #网络名：ssd_300_vgg   （基础网络时VGG，输入训练图像size是300x300）
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
        # Update feature shapes (try at least!) #下面这步我的理解就是让读者自行更改特征层的输入，未必论文中介绍的那几个block
        if update_feat_shapes:  #是否更新特征层图像尺寸？
            #可以理解为默认抽取特征的层是（4,7,8,9,10,11），也可以自己选择抽取的层，例如抽取6层，根据抽取层更新默认参数
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes) #预测的特征尺寸和给定的特征尺寸，输出更新后的特征图尺寸列表  
            self.params = self.params._replace(feat_shapes=shapes)#将更新的特征图尺寸shapes替换当前的特征图尺寸
        return r    #更新网络输入参数r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'): #定义权重衰减=0.0005，L2正则化项系数；数据类型是NHWC
        """Network arg_scope.
        """
        return ssd_arg_scope(weight_decay, data_format=data_format)

    def arg_scope_caffe(self, caffe_scope):
        """Caffe arg_scope used for weights importing.
        """
        return ssd_arg_scope_caffe(caffe_scope)

    # ======================================================================= #
    def update_feature_shapes(self, predictions): #更新特征形状尺寸（来自预测结果）
        """Update feature shapes from predictions collection (Tensor or Numpy
        array).
        """
        shapes = ssd_feat_shapes_from_net(predictions, self.params.feat_shapes)
        self.params = self.params._replace(feat_shapes=shapes)

    def anchors(self, img_shape, dtype=np.float32): #输入原始图像尺寸；返回每个特征层每个参考锚点框的位置及尺寸信息（x,y,h,w）
        """Compute the default anchor boxes, given an image shape.
        """
        # first->[((38*38*1),(38*38*1),(4*1),(4*1)),6]
        return ssd_anchors_all_layers(img_shape,  #这是个关键函数；检测所有特征层中的参考锚点框位置和尺寸信息
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)
                                      
                                      
    #编码，用于将标签信息，真实目标信息和锚点框信息编码在一起；得到预测真实框到参考框的转换值  
    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,  #未标注的标签（应该代表背景）
            ignore_threshold=0.5,     #IOU筛选阈值
            prior_scaling=self.params.prior_scaling, #特征图目标与参考框间的尺寸缩放（0.1,0.1,0.2,0.2）
            scope=scope)

    #解码，用锚点框信息，锚点框与预测真实框间的转换值，得到真是的预测框（ymin,xmin,ymax,xmax）  
    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    #通过SSD网络，得到检测到的bbox
    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip #选取top_k=400个框，并对框做修建（超出原图尺寸范围的切掉）
        rscores, rbboxes = \       #得到对应某个类别的得分值以及bbox
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \   #按照得分高低，筛选出400个bbox和对应得分
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \  #应用非极大值抑制，筛选掉与得分最高bbox重叠率大于0.5的，保留200个  
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        if clipping_bbox is not None:
            rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes   #返回裁剪好的bbox和对应得分
    #尽管一个ground?truth可以与多个先验框匹配，但是ground?truth相对先验框还是太少了，??
    #所以负样本相对正样本会很多。为了保证正负样本尽量平衡，SSD采用了hard?negative?mining，??
    #就是对负样本进行抽样，抽样时按照置信度误差（预测背景的置信度越小，误差越大）进行降序排列，??
    #选取误差的较大的top-k作为训练的负样本，以保证正负样本比例接近1:3??

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


#从预测类别概率层获取特征层的形状 predictions->[N,W,H,num_anchors,classes，6]
def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:  #l:是预测的特征形状[N,W,H,num_anchors,classes]
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray): #如果l是np.ndarray类型，则将l的形状赋给shape；否则将shape作为list 
            shape = l.shape  #[N,W,H,num_anchors,classes]
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]   #[W,H,num_anchors] ？？？？
        # Problem: undetermined shape...
        if None in shape:  #如果预测的特征尺寸未定，则使用默认的形状；否则将shape中的值赋给特征形状列表中
            return default_shapes #default_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        else:
            feat_shapes.append(shape)
    return feat_shapes    #返回更新后的特征尺寸list


#检测单个特征图中所有锚点的坐标和尺寸信息<相对于输入图>
def ssd_anchor_one_layer(img_shape, ,#原始图像shape,(300*300) 
                         feat_shape, #特征图shape   ,first->(38,38)
                         sizes,  #预设的box size,first->(21., 45.)
                         ratios, #aspect 比例,first->(2, .5)
                         step,   #anchor的层,first->(8)
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
    # Compute the position grid: simple way.              #计算中心点的归一化距离
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values... #归一化到原图的锚点中心坐标（x,y）;其坐标值域为(0,1) 
    """    
    #测试中，参数如下    
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
    #测试中，y和x的shape为（38,38）（38,38）               |   
    #-------------------------------------------|
    #y表示的图像的H，从上向下递增，同一行H相同                     |
    #y的值为                                       |
    #array([[ 0,  0,  0, ...,  0,  0,  0],      |
    #       [ 1,  1,  1, ...,  1,  1,  1],      |
    #       [ 2,  2,  2, ...,  2,  2,  2],      |
    #         ...,                              |
    #       [35, 35, 35, ..., 35, 35, 35],      |
    #       [36, 36, 36, ..., 36, 36, 36],      |
    #       [37, 37, 37, ..., 37, 37, 37]])     |
    #-------------------------------------------|
    #x表示的图像的W，从左向右递增，同一列W相同                     |
    #x的值为                                       |
    #array([[ 0,  1,  2, ...,  35,  36,  37],   |  
    #       [ 0,  1,  2, ...,  35,  36,  37],   | 
    #       [ 0,  1,  2, ...,  35,  36,  37],   | 
    #         ...,                              |
    #       [ 0,  1,  2, ...,  35,  36,  37],   | 
    #       [ 0,  1,  2, ...,  35,  36,  37],   |  
    #       [ 0,  1,  2, ...,  35,  36,  37]])  |
    #-------------------------------------------

    #对于第一个特征图（block4：38x38）；y=[[0,0,……0],[1,1,……1]，……[37,37,……，37]]；而x=[[0,1,2……，37]，[0,1,2……，37],……[0,1,2……，37]]  
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]  
    #测试中y=(y+0.5)×8/300,x=(x+0.5)×8/300
    #将38个cell对应锚点框的y坐标偏移至每个cell中心，然后乘以感受野，再除以原图  
    #可以得到在原图上，相对原图比例大小的每个锚点中心坐标x,y 
    #以第一个元素为例，简单方法为：（0+0.5)/38即为相对距离，而SSD-Caffe使用的是（0+0.5）*step/img_shape
    y = (y.astype(dtype) + offset) * step / img_shape[0]  #astype转换数据类型
    x = (x.astype(dtype) + offset) * step / img_shape[1]
    # Expand dims to support easy broadcasting. #扩展维度，维度变为（38,38,1），原维度为（38,38）
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)#该特征图上每个点对应的锚点框数量;如：对于第一个特征图每个点预测4个锚点框（block4：38x38），2+2=4  
    h = np.zeros((num_anchors, ), dtype=dtype) #对于第一个特征图，h的shape=4x；w的shape=4x 
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.（归一化）
    h[0] = sizes[0] / img_shape[0]  #第一个锚点框的高h[0]=起始锚点的高/原图大小的高；例如：h[0]=21/300 
    w[0] = sizes[0] / img_shape[1]  #第一个锚点框的宽w[0]=起始锚点的宽/原图大小的宽；例如：h[0]=21/300
    di = 1 #锚点宽个数偏移 
    if len(sizes) > 1: 
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]  #第二个锚点框的高h[1]=sqrt（起始锚点的高*起始锚点的宽）/原图大小的高；例如：h[1]=sqrt(21*45)/300  
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]  #第二个锚点框的高w[1]=sqrt（起始锚点的高*起始锚点的宽）/原图大小的宽；例如：w[1]=sqrt(21*45)/300  
        di += 1
    for i, r in enumerate(ratios): #遍历长宽比例，第一个特征图，r只有两个，2和0.5；共四个锚点宽size（h[0]~h[3]）  
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r) #例如：对于第一个特征图，h[0+2]=h[2]=21/300/sqrt(2);w[0+2]=w[2]=45/300*sqrt(2)  
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r) #例如：对于第一个特征图，h[1+2]=h[3]=21/300/sqrt(0.5);w[1+2]=w[3]=45/300*sqrt(0.5)   
    return y, x, h, w   #返回归一化的锚点坐标和尺寸 

#检测所有特征图中所有锚点的坐标和尺寸信息
def ssd_anchors_all_layers(img_shape,  #检测所有特征图中锚点框的四个坐标信息； 输入原始图大小[300*300]
                           layers_shape,   #每个特征层形状尺寸[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
                           anchor_sizes,   #起始特征图中框的长宽size anchor_sizes=[(21., 45.),(45., 99.),(99., 153.),(153., 207.),(207., 261.),(261., 315.)],
                           anchor_ratios,  #锚点框长宽比列表[[2, .5],    [2, .5, 3, 1./3], [2, .5, 3, 1./3],  [2, .5, 3, 1./3], [2, .5],  [2, .5]]
                           anchor_steps,   #锚点框相对原图缩放比例[8, 16, 32, 64, 100, 300]
                           offset=0.5,     #锚点中心在每个特征图cell中的偏移
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = [] #用于存放所有特征图中锚点框位置尺寸信息 
    for i, s in enumerate(layers_shape):  #6个特征图尺寸；如：第0个是38x38
        # first->anchor_bboxes=[y, x, h, w]=[(38*38*1),(38*38*1),(4*1),(4*1)]
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,    #分别计算每个特征图中锚点框的位置尺寸信息；
                                             anchor_sizes[i], #输入：第i个特征图中起始锚点框大小；如第0个是(21., 45.) 
                                             anchor_ratios[i],#输入：第i个特征图中锚点框长宽比列表；如第0个是[2, .5]
                                             anchor_steps[i], #输入：第i个特征图中锚点框相对原始图的缩放比；如第0个是8 
                                             offset=offset, dtype=dtype)#输入：锚点中心在每个特征图cell中的偏移
        layers_anchors.append(anchor_bboxes) #将6个特征图中每个特征图上的点对应的锚点框（6个或4个）保存
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
        # get_shape返回值，with_rank相当于断言assert，是否rank为指定值
        static_shape = x.get_shape().with_rank(rank).as_list()  #判断形状是不是rank维度
        # tf.shape返回张量，其中num解释为"The length of the dimension `axis`."，axis默认为0
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        # list，有定义的给数字，没有的给tensor
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

#每个特征层的位置偏移和类别置信度
def ssd_multibox_layer(inputs,  #输入特征层
                       num_classes,   #类别数    21
                       sizes,         #参考先验框的尺度 first->(21,45)
                       ratios=[1],    #默认的先验框长宽比为 first->[2,0.5,3,1/3]
                       normalization=-1, #默认不做正则化
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0: #如果输入整数，则进行L2正则化
        net = custom_layers.l2_normalization(net, scaling=True) #对通道所在维度进行正则化，随后乘以gamma缩放系数
    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)#每层特征图参考先验框的个数[4,6,6,6,4,4]

    # Location. #每个先验框对应4个坐标信息
    num_loc_pred = num_anchors * 4   #特征图上每个单元预测的坐标所需维度=锚点框数*4 
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,#通过对特征图进行3x3卷积得到位置信息
                           scope='conv_loc') #该部分是定位信息，输出维度为[N,特征图W,特征图H,每个单元所有锚点框坐标num_anchors * 4]  
    loc_pred = custom_layers.channel_to_last(loc_pred) # ensure data format be "NWHC"
    loc_pred = tf.reshape(loc_pred,  #最后整个特征图所有锚点框预测目标位置,tensor为[N，W，H, num_anchors，4]  
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.  #类别预测  
    num_cls_pred = num_anchors * num_classes #特征图上每个单元预测的类别所需维度=锚点框数*种类数  
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls') #该部分是类别信息，输出维度为[N,特征图W,特征图H,每个单元所有锚点框对应类别信息]  
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,  #将得到的feature maps reshape为[N，W，H, num_anchors，种类数]  
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    return cls_pred, loc_pred  #返回预测得到的类别和box位置 


#定义ssd网络结构
def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,      #分类数
            feat_layers=SSDNet.default_params.feat_layers,      #特征层 
            anchor_sizes=SSDNet.default_params.anchor_sizes,    
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,#正则化
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
    end_points = {} #用于收集每一层输出结果
    with tf.variable_scope(scope, 'ssd_300_vgg', [inputs], reuse=reuse): #tf.variable_scope()用来指定变量的作用域，reuse设置变量的重用，参考https://blog.csdn.net/wanglitao588/article/details/76976428
        # Original VGG-16 blocks.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1') #VGG16网络的第一个conv，重复2次卷积，核为3x3,64个特征  
        end_points['block1'] = net     #conv1_2结果存入end_points，name='block1'  
        net = slim.max_pool2d(net, [2, 2], scope='pool1')  #池化层
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')  #重复2次卷积，核为3x3,128个特征  
        end_points['block2'] = net   #conv2_2结果存入end_points，name='block2'  
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')  #重复3次卷积，核为3x3,256个特征 
        end_points['block3'] = net   #conv3_3结果存入end_points，name='block3'  
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')  #重复3次卷积，核为3x3,512个特征,输出[batch,38,38,512]
        end_points['block4'] = net   #conv4_3结果存入end_points，name='block4'  
        net = slim.max_pool2d(net, [2, 2], scope='pool4') 
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')   #重复3次卷积，核为3x3,512个特征
        end_points['block5'] = net   #conv5_3结果存入end_points，name='block5'  
        net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

        # Additional SSD blocks. #外加的SSD层
        # Block 6: let's dilate the hell out of it! #去掉了VGG的全连接层 
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6') #将VGG基础网络最后的池化层结果做扩展卷积（带孔卷积）；  
        end_points['block6'] = net   #conv6结果存入end_points，name='block6'  
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training) #dropout层 
        # Block 7: 1x1 conv. Because the fuck.
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7') #将dropout后的网络做1x1卷积，输出1024特征，name='block7'，输出[batch,19,19,1024] 
        end_points['block7'] = net  
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training) #将卷积后的网络继续做dropout

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'  #对上述dropout的网络做1x1卷积，然后做3x3卷积，,输出512特征图，name=‘block8’,输出[batch,10,10,512]  
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID') 
        end_points[end_point] = net
        end_point = 'block9'  #对上述网络做1x1卷积，然后做3x3卷积，输出256特征图，name=‘block9’,输出[batch,5,5,256] 
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block10' #对上述网络做1x1卷积，然后做3x3卷积，输出256特征图，name=‘block10’,输出[batch,3,3,256]   
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block11' #对上述网络做1x1卷积，然后做3x3卷积，输出256特征图，name=‘block11’，输出[batch,1,1,256]   
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net

        # Prediction and localisations layers. #预测和定位 
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):  #遍历特征层
            with tf.variable_scope(layer + '_box'): #起个命名范围
                # p->[N，W，H, num_anchors，种类数,6层] 
                p, l = ssd_multibox_layer(end_points[layer],#做多尺度大小box预测的特征层，返回每个cell中每个先验框预测的类别p和预测的位置l  
                                          num_classes,      #种类数
                                          anchor_sizes[i],  #先验框尺度（同一特征图上的先验框尺度和长宽比一致） 
                                          anchor_ratios[i], #先验框长宽比
                                          normalizations[i])#每个特征正则化信息，目前是只对第一个特征图做归一化操作
            #把每一层的预测收集
            predictions.append(prediction_fn(p))#prediction_fn为softmax，预测类别概率 （损失函数中的-c_i^p）
            logits.append(p)  #把每个cell每个先验框预测的类别的概率值存在logits中 (损失函数中的c_i^p)
            localisations.append(l) #预测位置信息

        #所属某个类别的概率，位置预测结果，返回类别预测结果置信度，以及特征层  
        return predictions, localisations, logits, end_points
ssd_net.default_image_size = 300

#用于约束网络中的超参数设定，设置默认参数，即我们不设置函数的参数时，默认用这里的设置
#参看：https://blog.csdn.net/u013921430/article/details/80915696
#      https://blog.csdn.net/DeepOscar/article/details/82762929?utm_source=blogxgwz3
def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'): #权重衰减系数=0.0005；其是L2正则化项的系数
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    # 这个函数的作用是给list_ops中的内容设置默认值。但是每个list_ops中的每个成员需要用@add_arg_scope修饰才行。
    # 并不是所有的方法都能用arg_scope设置默认参数, 只有用@slim.add_arg_scope修饰过的方法才能使用arg_scope
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
#损失函数定义为位置误差和置信度误差的加权和；假设某一layer预测类别概率形状为S1->[N,H,W,anchorNum,21]
def ssd_losses(logits,        #预测类别概率[S1,S2,S3,S4,S5,S6]
               localisations, #预测位置[(N,H,W,anchorNum,4),6层)]
               gclasses,      #ground truth 与先验框的类别[(N,W,H,anchorNum), 6]
               glocalisations,#ground truth 与先验框的位置偏移 [(N,W,H,anchorNum,4), 6]
               gscores,       #ground truth 与先验框的交并比得分（置信度）[(N,W,H,anchorNum), 6]
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,     #位置误差权重系数
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
        for i in range(len(logits)): # len(logits) = 6个feature map
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))  #将预测类别的概率值reshape成（N*H*W*anchorNum,21）
            fgclasses.append(tf.reshape(gclasses[i], [-1]))           #真实类别（N*H*W*anchorNum,）
            fgscores.append(tf.reshape(gscores[i], [-1]))             #真实目标的得分（N*H*W*anchorNum,）
            flocalisations.append(tf.reshape(localisations[i], [-1, 4])) #预测目标边框坐标 （N*H*W*anchorNum,4）
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4])) #用于将真实目标gt的坐标进行编码存储 
        # And concat the crap! 拼接，例如原来是[[S1],[S2],[S3],[S4],[S5],[S6]]->[S1,S2,S3,S4,S5,S6]
        logits = tf.concat(flogits, axis=0)   # 预测概率[(N*H*W*anchorNum,21),6]
        gclasses = tf.concat(fgclasses, axis=0) #先验框类别 [(N*H*W*anchorNum,),6]
        gscores = tf.concat(fgscores, axis=0)   #先验框与真实框IOU(置信度) [(N*H*W*anchorNum,),6]
        localisations = tf.concat(flocalisations, axis=0)   # 预测位置[(N*H*W*anchorNum,4),6]
        glocalisations = tf.concat(fglocalisations, axis=0) # 先验框位置[(N*H*W*anchorNum,4),6]
        dtype = logits.dtype

        # Matching strategy,Compute positive matching mask...
        pmask = gscores > match_threshold #预测框与真实框IOU>0.5则将这个先验作为正样本
        fpmask = tf.cast(pmask, dtype)  #正样本标志
        n_positives = tf.reduce_sum(fpmask) #正样本数目

        # Hard negative mining...
        #为了保证正负样本尽量平衡，SSD采用了hard negative mining，就是对负样本进行抽样，
        #抽样时按照置信度误差（预测背景的置信度越小，误差越大）进行降序排列，
        #选取误差的较大的top-k作为训练的负样本，以保证正负样本比例接近1:3  
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)   #类别预测[(N*H*W*anchorNum,21),6]
        nmask = tf.logical_and(tf.logical_not(pmask),gscores > -0.5) #负样本
        fnmask = tf.cast(nmask, dtype) #负样本标志
        nvalues = tf.where(nmask,  #负样本置信度 [(N*H*W*anchorNum,),6]
                           predictions[:, 0], #背景的置信度，即21个类别中第0个背景类
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1]) #变成一维(N*H*W*anchorNum*6,)
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32) #负样本总数目
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size#负样本数量，保证是正样本3倍 
        n_neg = tf.minimum(n_neg, max_neg_entries)  #保证选取的负样本数量不超过总的负样本数量

        #这里注意是-nvalues_flat，得到的val也是负数，负数降序排列，相当于正数升序
        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)  #抽样时按照置信度误差（预测背景的置信度越小，误差越大）进行降序排列，选取误差的较大的top-k作为训练的负样本  
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype) #负样本标志

        # Add cross-entropy loss.  #交叉熵（计算L_conf）
        with tf.name_scope('cross_entropy_pos'): #正样本
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,   #预测类别置信度logits（c_i^p）->[(N*H*W*anchorNum,21),6]
                                                                  labels=gclasses) #gclasses->[(N*H*W*anchorNum,),6]
            loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value') #将置信度误差乘以正样本数后除以batch-size  
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_neg'): #负样本
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)
            loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        # Add localization loss: smooth L1, L2, ... #(计算L_loc)
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localisations - glocalisations) #预测位置偏移-真实位置偏移值；然后做Smooth L1 loss  
            loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')#将上面的loss*权重（=alpha*正样本数）求和后除以batch-size  
            tf.losses.add_loss(loss) #获得置信度误差和位置误差的加权和


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
