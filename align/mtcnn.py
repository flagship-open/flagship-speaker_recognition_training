# from detector.align import detect_face, facenet
# from .align import detect_face, facenet
import align.detect_face as detect_face
import align.facenet as facenet
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
from scipy import misc
import sys,cv2
import os,pathlib, glob
import random
from time import sleep

def mtcnn_img(img):

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    margin = 0
    img = np.array(img)
    if img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:, :, 0:3]
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    # print(bounding_boxes)
    det  =bounding_boxes[:,0:4]
    x_center = img_size[1] / 2
    y_center = img_size[0] / 2
    idx = 0
    d = 999999

    for num2,i in enumerate(det):
        l, t, r, b = i
        c_d = ((x_center -(l+r)/2)+(y_center-(t+b)/2))
        if (abs((x_center -(l+r)/2))+abs((y_center-(t+b)/2)))<=d:
            idx = num2
            d = c_d
        # print(img_path)
        # print(det)
    if len(det)!=0:
        l,t,r,b = det[idx]
        # l = int(l)
        # t  = int(t)
        # r = int(r)
        # b = int(b)
        l = int(np.maximum(l-margin/2, 0))
        t = int(np.maximum(t-margin/2, 0))
        r = int(np.minimum(r+margin/2, img_size[1]))
        b = int(np.minimum(b+margin/2, img_size[0]))
        cropped = img[t:b,l:r]
        scaled = cv2.resize(cropped, (112, 112))
    else:
        l = int(img_size[1] * 0.05)
        t = int(img_size[0] * 0.05)
        r = int(img_size[1] * 0.95)
        b = int(img_size[0] * 0.95)
        cropped = img[t:b, l:r]
        scaled = cv2.resize(cropped, (112, 112))

    return scaled
