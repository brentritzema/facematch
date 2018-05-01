import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
import argparse
import os
import os.path


parser = argparse.ArgumentParser()
parser.add_argument("--img", type = str, required=True)
args = parser.parse_args()

# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160

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

def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
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

def getEmbedding(resized):
    reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding

def compare2face(img1,img2):
    face1 = getFace(img1)
    face2 = getFace(img2)
    if face1 and face2:
        # calculate Euclidean distance
        dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
        return dist
    return -1

def generateDictionary():
    for filename in os.scandir("authorized"):
        if (filename.name.endswith(".png") or filename.name.endswith(".jpg")) and filename.is_file():
            name = os.path.splitext(filename.name)[0]
            face = getFace(cv2.imread("authorized/" + filename.name))
            if face:
                authorized[name] = face[0]


def findMatch(unknown_img):
    unknown = getFace(unknown_img)
    max_dist = 1.10
    match = "No match"
    if unknown:
        unknown_embeddings = unknown[0]['embedding']
        for name in authorized:
            dist = np.sqrt(np.sum(np.square(np.subtract(authorized[name]['embedding'], unknown_embeddings))))
            if dist < max_dist:
                max_dist = dist
                match = name
            print(name, dist)
    else:
        match = "No face found"
    return match

generateDictionary()
img = cv2.imread(args.img)
match = findMatch(img)
print(match)
