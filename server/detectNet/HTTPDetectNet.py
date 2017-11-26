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

from digits_detect import DIGITS_Handler
import numpy as np
import scipy.io as sio
import scipy.misc
import caffe, os, sys, cv2

from BaseHTTPServer import BaseHTTPRequestHandler
from BaseHTTPServer import HTTPServer
import cgi
import json

CLASSES = ('yoshiki','weiliang','soga','shudo','ohnishi',
            'nakajima', 'ken', 'aoki', 'kanda','iwasaki')

class HTTPDetectNetHandler(BaseHTTPRequestHandler):  
    digits_handler = DIGITS_Handler()

    def detect(self, im):
        # Visualize detections for each class
        ver_scale = 384.0 / im.shape[0]
        hor_scale = 1248.0 / im.shape[1]
        detectNet_in = scipy.misc.imresize(im, (384,1248), 'bilinear')
        self.digits_handler.set_image(detectNet_in)
        score = self.digits_handler.run()[0][0]
        label_names = []
        pred_rects = []
        pred_probs = []
        for cls_ind, cls in enumerate(CLASSES[:]):
            bbox_list = score['bbox-list-class' + str(cls_ind)]
            bbox_list = bbox_list[0]
            nonzero_inds = np.nonzero(bbox_list[:,-1])
            for index in nonzero_inds[0]:
                label_names.append(cls)
                pred_rects.append([int(bbox_list[index, 0]/hor_scale),
                                    int(bbox_list[index, 1]/ver_scale),
                                    int(bbox_list[index, 2]/hor_scale),
                                    int(bbox_list[index, 3]/ver_scale),
                                    ])
                pred_probs.append(float(bbox_list[index, -1]))
            cls_ind += 1
        obj = {
            'class_names' : label_names,
            'points' : pred_rects,
            'probability' : pred_probs}
        print(obj)
        json_data = json.dumps(obj).encode('utf-8')
        return json_data

    def do_GET(self):
        body = 'test message'
        body_bytes = bytes(body)
        self.wfile.write(body_bytes)


    def do_POST(self):
        ''' old 2017/11/26
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers = self.headers,
            environ={'REQUEST_METHOD':'POST'})
        rcv_item = form['file']
        rcv_data = rcv_item.value
        '''
        rcv_item = self.rfile
        rcv_len = int(self.headers.get('Content-Length'))
        rcv_data = rcv_item.read(rcv_len)

        narray=np.fromstring(rcv_data,dtype=np.uint8)
        image = cv2.imdecode(narray, cv2.CV_LOAD_IMAGE_COLOR)
        print('image shape length : ' + str(len(image.shape)))
        print('height: ' + str(image.shape[0]))
        print('width: ' + str(image.shape[1]))
        print('channels: ' + str(image.shape[2]))
        print('dtype: ' + str(image.dtype))
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        output_json = self.detect(image)
        self.send_response(200) #200 is http error code meaning 'OK'
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(output_json)
 

if __name__ == '__main__':
    HOST = '192.168.0.24'
    PORT = 5000
    server = HTTPServer((HOST, PORT), HTTPDetectNetHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.shutdown()

