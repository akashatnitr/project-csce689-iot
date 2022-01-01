# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
import urllib.request
import time
import math
import numpy as np
from socketIO_client import SocketIO, LoggingNamespace


#server macbook 10.0.0.95
host = "10.0.0.95"
# construct the argument parser and parse the arguments

def on_bbb_response(*args):
    print('on_bbb_response', args)
    # socketIO.emit('message', {'second_msg': 'teset'}, on_bbb_response)


known_loc = './inferences/known_faces/'
unknown_loc = './inferences/unknown_faces/'

ap = argparse.ArgumentParser()
ap.add_argument("-e",
                "--encodings",
                required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=False, help="path to input image")
ap.add_argument("-u", "--url", required=True, help="path to input image")
ap.add_argument("-d",
                "--detection-method",
                type=str,
                default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())
if (args['url'] is not None):
    url = args["url"]
    urllib.request.urlretrieve(url, "local-filename.jpg")
    time.sleep(1.0)

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load the input image and convert it from BGR to RGB

# image = cv2.imread(args["image"])
image = cv2.imread('./local-filename.jpg')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []
print('connecting with sockeet io')
with SocketIO(host, 5000, LoggingNamespace) as socketIO:
    print('connected with sockeet io')
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
                socketIO.emit('message', {'xxx': 'yyy'}, on_bbb_response)
                socketIO.wait_for_callbacks(seconds=1)
       

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
    # draw the predicted face name on the image
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 255, 0), 2)

    if (name == 'Unknown'):
        #save the faces to the unknown folder
        padding = 0.15
        padding_down = 0.4
        t = math.floor(max(0, top - padding * (bottom - top)))
        b = math.floor(
            min(np.size(image, 0),
                bottom + padding_down * (bottom - top)))  #height
        l = math.floor(max(0, left - padding * (right - left)))
        r = math.floor(min(np.size(image, 1),
                           right + padding * (right - left)))  #width
        unknown_img = image[t:b, l:r]
        # print(left, right, top, bottom)
        cv2.imshow("Image", unknown_img)
        cv2.waitKey(0)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

#ssd for the complete human of the known and unknown face if needed