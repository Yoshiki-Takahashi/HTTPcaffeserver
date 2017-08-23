# -*- coding:utf-8 -*-
#!/usr/bin/python
#from http.client import HTTPConnection
from httplib import HTTPConnection
import json
import cv2
import urllib

class MyHTTPClient:
  def __init__(self):
    self.serverIP = '192.168.0.28'#'131.112.51.42'
    self.serverPort = 5001
    self.connection = HTTPConnection(self.serverIP, self.serverPort)

  def send_image(self, image):
    #image = cv2.imread(path,cv2.IMREAD_COLOR)
    #convert image and make body data
    rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    retval,array = cv2.imencode('.png',rgb_im)
    buf = array.tobytes()
    body = urllib.urlencode({'file': buf})

    #response = requests.post(self.URL, data=body)
    self.connection.request('POST', '', body)
    response = self.connection.getresponse()
    json_data = json.loads(response.read().decode('utf-8'))
    return json_data

class Camera_Classifier:
  def __init__(self):
    self.http_client = MyHTTPClient()

  def run(self):
    capture = cv2.VideoCapture(1)
    if capture.isOpened() is False:
      raise("IO Error")

    cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
    while True:
      ret, image = capture.read()
      if ret == False:
        continue
      json_data = self.http_client.send_image(image)
      face_names = json_data['class_names']
      face_pt = json_data['face_points']
      for fn, fp in zip(face_names, face_pt):
        cv2.putText(image, fn, tuple(fp[:2]), 0, 2.0, (255,255,255))
        cv2.rectangle(image, tuple(fp[:2]), tuple(fp[2:4]), (255,255,255))

      cv2.imshow("Capture", image)
      cv2.waitKey(10)

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

