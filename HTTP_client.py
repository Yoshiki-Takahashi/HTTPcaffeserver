# -*- coding:utf-8 -*-
#!/usr/bin/python
#from http.client import HTTPConnection
from httplib import HTTPConnection
import cv2

class MyHTTPClient:
  def __init__(self):
    self.serverIP = '192.168.0.28'
    self.serverPort = 5001
    self.connection = HTTPConnection(self.serverIP, self.serverPort)

  def send_image(self, path):
    image = cv2.imread(path,cv2.IMREAD_COLOR)
    retval,buf = cv2.imencode('.png',image)
    self.connection.request('POST', '', buf)
    response = self.connection.getresponse()
    print(response.read())
    '''
    #with open(path, 'rb') as image:
      data = 'client send message'
      send_bytes = bytes(data, 'utf-8')
      self.connection.request('POST', '', send_bytes)
      response = self.connection.getresponse()
      print(response.read())
    '''


if __name__ == '__main__':
  client = MyHTTPClient()
  client.send_image('../img8492-0.png')

