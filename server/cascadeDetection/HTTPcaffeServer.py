# -*- coding:utf-8 -*-
#!/usr/bin/python2
from BaseHTTPServer import BaseHTTPRequestHandler
from BaseHTTPServer import HTTPServer
import cgi
import numpy as np 
import json
import cv2
from digits_classify import DIGITS_Handler

class HTTPCaffeHandler(BaseHTTPRequestHandler):  
    caffe_handler = DIGITS_Handler()

    def get_face_rec(self, image):
        cascade_path = '/home/ytakahashi/HTTPcaffeserver/haarcascade_frontalface_alt.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        color = (255,255,255)
        facerect = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(1,1))
        max_factor = 1.7
        rect_pts = []
        for rect in facerect:
            edge_len = float(rect[2])
            center_pt = [rect[0] + rect[2]/2, rect[1] + rect[3]/2]
            Y,X = image.shape[:2]
            factor = min(max_factor,
              2*center_pt[0]/edge_len,
              2*center_pt[1]/edge_len,
              2*(X - center_pt[0])/edge_len,
              2*(Y - center_pt[1])/edge_len)
            tlx = int(center_pt[0] - factor*edge_len/2)
            tly = int(center_pt[1] - factor*edge_len/2)
            brx = int(center_pt[0] + factor*edge_len/2)
            bry = int(center_pt[1] + factor*edge_len/2)
            rect_pts.append([tlx,tly,brx,bry])
        return rect_pts

    def do_GET(self):
        body = 'test message'
        body_bytes = bytes(body, 'utf-8')
        self.wfile.write(body_bytes)

    def do_POST(self):
        # this gets RGB, png format!!!
        # data is in 'file' field
        ##### recieve data ######
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
        ##### equalize image and predict #####
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        face_rec_list = self.get_face_rec(image)
        face_name_list = []
        prob_list = []
        for fr in face_rec_list:
            self.caffe_handler.set_image(img_output[fr[1]:fr[3], fr[0]:fr[2]])
            output = self.caffe_handler.run()
            face_name_list.append(output[0][0])
            prob_list.append(output[0][1])
        ##### sort list    #####
        '''
        prob_sort = np.argsort(prob_list)
        name_sort = [face_name_list[i] for i in prob_sort]
        rec_sort  = [face_rec_list[i] for i in prob_sort]
        print(prob_sort)
        '''
        ##### make headers #####
        self.send_response(200) #200 is http error code meaning 'OK'
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        ##### make body data #####
        obj = {
          'class_names' : face_name_list,
          'face_points' : face_rec_list,
          'probability' : prob_list}
        json_data = json.dumps(obj).encode('utf-8')
        self.wfile.write(json_data)
        return

if __name__ == '__main__':
    HOST = '192.168.0.28'
    PORT = 5001
    handler = DIGITS_Handler()
    server = HTTPServer((HOST, PORT), HTTPCaffeHandler)
    try:
        server.serve_forever()  
    except KeyboardInterrupt:
        pass
    server.shutdown()
