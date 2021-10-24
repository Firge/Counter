import PIL.Image
import PIL.ImageDraw
import face_recognition as fr
import matplotlib.pyplot as plt
import cv2
import os
import urllib.request
import numpy as np

faces_images = []
for i in os.listdir('faces/'):
    faces_images.append(fr.load_image_file('faces/' + i))
known_face_encodings = []
for i in faces_images:
    known_face_encodings.append(fr.face_encodings(i)[0])
known_face_names = []
for i in os.listdir('faces/'):
    i = i.split('.')[0]
    known_face_names.append(i)
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
req = open('photos/bb.jpg')
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
frame = cv2.imdecode(arr, -1)
