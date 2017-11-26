# -*- coding:utf-8 -*-
#!/usr/bin/python
#from http.client import HTTPConnection
from httplib import HTTPConnection
import json
import cv2
import urllib
import scipy.misc

class MyHTTPClient:
  def __init__(self):
    #self.serverIP = '192.168.0.24'
    self.serverIP = '131.112.51.42'
    self.serverPort = 5004
    self.connection = HTTPConnection(self.serverIP, self.serverPort)

  def send_image(self, image):
    #image = cv2.imread(path,cv2.IMREAD_COLOR)
    #convert image and make body data
    rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    retval,array = cv2.imencode('.png',rgb_im)
    buf = array.tobytes()
    #body = urllib.urlencode({'file': buf})

    self.connection.request('POST', '', buf)
    response = self.connection.getresponse()
    json_data = json.loads(response.read().decode('utf-8'))
    return json_data

class Camera_Classifier:
  def __init__(self):
    self.http_client = MyHTTPClient()

  def run(self):
    capture = cv2.VideoCapture(0)
    if capture.isOpened() is False:
      raise("IO Error")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('memberDetection.avi', fourcc, 5.0, (640,480),True)

    cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
    while capture.isOpened():
      ret, image = capture.read()
      if ret == False:
        continue
      json_data = self.http_client.send_image(image)
      face_names = json_data['class_names']
      face_pt = json_data['points']
      for fn, fp in zip(face_names, face_pt):
        cv2.putText(image, fn, tuple(fp[:2]), 0, 1.0, (255,255,255))
        cv2.rectangle(image, tuple(fp[:2]), tuple(fp[2:4]), (255,255,255))

      print(image.shape)
      out.write(image)
      cv2.imshow("Capture", image)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    capture.release()
    out.release()

if __name__ == '__main__':
  '''
  client = MyHTTPClient()
  image = cv2.imread('../lena.png',1)
  json_data = client.send_image(image)
  if len(json_data['class_names']) > 0:
    print(json_data['class_names'][0])
  '''


  camera = Camera_Classifier()
  camera.run()

