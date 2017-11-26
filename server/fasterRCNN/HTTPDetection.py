#!/usr/bin/python

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
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2

from BaseHTTPServer import BaseHTTPRequestHandler
from BaseHTTPServer import HTTPServer
import cgi
import json

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

cfg.TEST.HAS_RPN = True  # Use RPN for proposals
prototxt = os.path.join(cfg.MODELS_DIR, NETS['vgg16'][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS['vgg16'][1])

if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

caffe.set_mode_gpu()
caffe.set_device(0)
cfg.GPU_ID = 0
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

class HTTPCaffeHandler(BaseHTTPRequestHandler):  
    '''
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    prototxt = os.path.join(cfg.MODELS_DIR, NETS['vgg16'][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS['vgg16'][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    '''

    def detect(self, im):
        # Detect all object classes and regress object bounds
        scores, boxes = im_detect(net, im)

        # Visualize detections for each class
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        label_names = []
        pred_rects = []
        pred_probs = []
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            for i in inds:
                label_names.append(cls)
                pred_rects.append([ int(pt) for pt in dets[i, :4]])
                pred_probs.append(dets[i, -1].item(0))
        obj = {
            'class_names' : label_names,
            'points' : pred_rects,
            'probability' : pred_probs}
        json_data = json.dumps(obj).encode('utf-8')
        return json_data

    def do_GET(self):
        body = 'test message'
        body_bytes = bytes(body)
        self.wfile.write(body_bytes)


    def do_POST(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers = self.headers,
            environ={'REQUEST_METHOD':'POST'})
        rcv_item = form['file']
        rcv_data = rcv_item.value
        narray=np.fromstring(rcv_data,dtype=np.uint8)
        image = cv2.imdecode(narray, cv2.CV_LOAD_IMAGE_COLOR)
        print('image shape length : ' + str(len(image.shape)))
        print('width: ' + str(image.shape[0]))
        print('height: ' + str(image.shape[1]))
        print('channels: ' + str(image.shape[2]))
        print('dtype: ' + str(image.dtype))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        output_json = self.detect(image)
        self.send_response(200) #200 is http error code meaning 'OK'
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(output_json)
 

if __name__ == '__main__':
    HOST = '192.168.0.24'
    PORT = 5000
    server = HTTPServer((HOST, PORT), HTTPCaffeHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.shutdown()

