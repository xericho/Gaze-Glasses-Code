#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import csv

# create writer object for csv
f = open('gaze_objects.csv','wb')
writer = csv.writer(f)
writer.writerow(['object',' xmin',' ymin',' width',' height'])

#foldername = 'Test_Images_One_Object'
foldername = 'ohno'

CONF_THRESH = 0.3

CLASSES = ('__background__', # always index 0
            'brain', 'star', 'lunch', 'park',
            'shark', 'skeleton', 'frame')

NETS = {'gaze': ('gaze_model_faster_rcnn_final.caffemodel')	}

def vis_detections(ax, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    #print 'inds: {}'.format(inds)
    if len(inds) == 0:
        return

    # get the highest confidence score
    conf = []
    for i in inds:
        conf.append(dets[i, -1])
    #print 'conf: {}'.format(conf)
    m = max(conf)
    maxConf = [i for i, j in enumerate(conf) if j == m]

    #maxConf = conf.index(max(conf))
    #print 'maxConf: {}'.format(maxConf)

    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(10, 10))
    #ax.imshow(im, aspect='equal')
    for i in maxConf:
        bbox = dets[i, :4]
        #print 'bbox: {}'.format(bbox)
        score = dets[i, -1]
        #print 'score: {}'.format(score)

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
	print  class_name, score, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
	text = '{},{},{},{},{},'.format(class_name, int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1]))

        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'P({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)

    return text
    #plt.axis('off')
    #plt.tight_layout()
    #plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, foldername, image_name)
    im = cv2.imread(im_file)

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(im, aspect='equal')

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # variable for storing data to write to csv file
    dataObject = ''

    # Visualize detections for each class
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        #print 'cls_ind: {}'.format(cls_ind)
        #print 'cls: {}'.format(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        #print 'cls_boxes: {}'.format(cls_boxes)
        cls_scores = scores[:, cls_ind]
        
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        #print 'dets: {}'.format(dets)
        keep = nms(dets, NMS_THRESH)
        #print 'keep: {}'.format(keep)
        dets = dets[keep, :]
        #print 'dets: {}'.format(dets)
        text = vis_detections(ax, im, cls, dets, thresh=CONF_THRESH)
        if text != None:
            dataObject += text

    if dataObject == '':
        writer.writerow('')
    else:
        writer.writerow(dataObject.split(','))

    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='gaze')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR,'gaze_model', 
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              'gaze_model_faster_rcnn_final_v3.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    """
    im_names = ['test1.png','lunch_1804.jpg','park_0400.jpg',
                'test2.png','test3.png','trainertestimage.jpg',
                'test4.png','test5.png','test6.png','test7.png']
    """
    # gets the name of directory and puts it in a list like ^
    im_names = os.listdir(os.path.join(cfg.DATA_DIR, foldername))
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)
        demo(net, im_name)

    f.close()
    plt.show()
