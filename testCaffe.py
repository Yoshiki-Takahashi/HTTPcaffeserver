from TCPcaffeServer import MyCaffeHandler
from digits_classify import DIGITS_Handler
import skimage.io
import cv2
import os

caffe_handler = DIGITS_Handler()

imagedir = '/home/user/Documents/tcpSocket/image/'
imagelist = os.listdir(imagedir)

for image_name in imagelist:
  image = cv2.imread(imagedir+image_name)
  rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  caffe_handler.set_image(rgb_im)
  print(caffe_handler.run()[0][0])

