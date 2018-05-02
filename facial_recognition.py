# facial_recognition.py is an adaption of the work from FaceNet by David Sandberg and a use-case of it by Arun Mandal
# This program simply functions as a server, which subscribes to an MQTT topic to receive images. Once an image is
# received it is run through the system which detects faces and then tries to recognize them. Once a match is found (or
# no match is found) it publishes to a seperate topic which the RasperryPi is subscribed to so that the Pi can act on the
# information.

# Authors: Toussaint Cruise, Brent Ritzema

import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
import argparse
import os
import os.path
import paho.mqtt.client as mqtt
import time

BROKER = "iot.cs.calvin.edu"
PORT = 1883
QOS = 1

IMAGE_TOPIC='highmount/camera'
IMAGE_FILE_NAME='last.jpg'
AUTHORIZED_FOLDER_NAME='authorized'

AUTH_THRESHOLD=1.02

#This function is called when a message is recieved
#It takes the image from the PI, writes it to a file, and then send it through the facial recognition system
def on_message(mosq, obj, msg):                             # Function to retrieve file when received
  print("received image")
  with open(IMAGE_FILE_NAME, 'wb') as fd: #overwrites any old file
      fd.write(msg.payload)
  img = cv2.imread(IMAGE_FILE_NAME)
  match = findMatch(img)
  print(match) 

client = mqtt.Client("P2")                                # Start MQTT Client, NB changed to P2
client.connect(BROKER, PORT, 60)                 # Connect to server
client.subscribe(IMAGE_TOPIC, QOS)

client.on_message = on_message                            # This is key - it calls the function



# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160

#setting up the tensorflow session (facenet useses it)
sess = tf.Session()

# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

# read model file downloaded from https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view
facenet.load_model("20180402-114759/20180402-114759.pb")

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

#dictionary of faces that are authorized
authorized = {}

# This function is the face detection and embedding function. It takes an image, detects all the faces within it, 
# finds embeddings for every face, and returns a list of the faces via their corresponding embeddings.
def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            #A lot of values and calculations here are just left as default for face detection
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]],'embedding':getEmbedding(prewhitened)})
    return faces

# This function is what allows for recognition. By generating values for particular features of a face, it allows
# comparisons of those features to other faces.
def getEmbedding(resized):
    reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding

# This function generates a python dictionary of names and their embeddings found in the authorized
# folder. This allows a comparison of the received image to a set of authorized users. In order to be
# authorized you only need to put one image of yourself in the authorized folder with the title as your name.
def generateDictionary():
    for filename in os.scandir(AUTHORIZED_FOLDER_NAME):
        if (filename.name.endswith(".png") or filename.name.endswith(".jpg")) and filename.is_file():
            name = os.path.splitext(filename.name)[0]
            face = getFace(cv2.imread(AUTHORIZED_FOLDER_NAME + "/" + filename.name))
            if face:
                authorized[name] = face[0]


# This function compares an image to all the authorized images. If the images matches another
# passed the empirically found 'threshold of matching', then the name of the user is returned,
# otherwise the appropriate output is returned (no match or no face found)
def findMatch(unknown_img):
    unknown = getFace(unknown_img)
    max_dist = AUTH_THRESHOLD #threshold
    match = "No match" #default if none is found
    if unknown:
        unknown_embeddings = unknown[0]['embedding']
        for name in authorized:
            #compare embeddings
            dist = np.sqrt(np.sum(np.square(np.subtract(authorized[name]['embedding'], unknown_embeddings))))
            #find person who is most likely if multiple below threshold
            if dist < max_dist:
                max_dist = dist
                match = name
            print(name, dist)
    else:
        match = "No face found"
    return match

generateDictionary()
while True:                                               # Loop and wait for next image
   client.loop(10)

