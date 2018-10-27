# -*- coding:utf-8 -*-

import os
import tensorflow as tf
import matplotlib.image as mpimg
from nets import ssd_vgg_300, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Python解释器会搜索当前目录、所有已安装的内置模块和第三方模块，搜索路径存放在sys模块的path变量中
# append添加自己的搜索目录
sys.path.append('../')

slim = tf.contrib.slim

# 使用tf.InteractiveSession()来构建会话的时候，我们可以先构建一个session然后再定义操作（operation），
# 如果我们使用tf.Session()来构建会话我们需要在会话构建之前定义好全部的操作（operation）然后再构建会话。
isess = tf.InteractiveSession()

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))


# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(  #预处理
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0) #扩展维度[h,w,c]->[1,h,w,c]

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None  #locals() 函数会以字典类型返回当前位置的全部局部变量。
ssd_net = ssd_vgg_300.SSDNet()  #新建ssd模型对象
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = 'checkpoints/ssd_300_vgg/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

# Main image processing routine.
# 参看：https://drive.google.com/file/d/164mVbMBhoMzY5pkaEOdK3IIcIwTOj2B-/view
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    # 求解图像[N,H,W,C]，预测概率[N,H,W,anchorNum,4,6层]，预测位置[N,H,W,anchorNum,21,6层]，groundTruth boxes
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    # [batch,H*W*anchorNum,大于阈值的类别],[batch,H*W*anchorNum,大于阈值的类别得分]，[batch,H*W*anchorNum,大于阈值的类别位置]
    # 简单理解就是每一个点对应21个类别得分，我们把类别去掉背景后剩下20个类别从中找到大于阈值的类别保留下来。
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# Test on some demo image and visualize output.
path = os.getcwd()+'/demo'
image_names = sorted(os.listdir(path))
img = mpimg.imread(path +"/"+ image_names[-1])
rclasses, rscores, rbboxes =  process_image(img)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

