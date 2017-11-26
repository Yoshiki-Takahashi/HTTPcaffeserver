
from httplib import HTTPConnection
import json
import cv2
import urllib
import scipy.misc

class MyHTTPClient:
  def __init__(self):
    self.serverIP = '192.168.0.24'
    #self.serverIP = '131.112.51.42'
    self.serverPort = 5000
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
    image = cv2.imread('/home/yoshiki/Documents/tcpSocket/python/images/OutVideopi03_14533.jpg')

    cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
    while True:
      json_data = self.http_client.send_image(image)
      face_names = json_data['class_names']
      face_pt = json_data['points']
      for fn, fp in zip(face_names, face_pt):
        cv2.putText(image, fn, tuple(fp[:2]), 0, 1.0, (255,255,255))
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

