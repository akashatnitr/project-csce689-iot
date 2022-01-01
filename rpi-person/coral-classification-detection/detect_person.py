# USAGE
# python detect_video.py --model mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --labels mobilenet_ssd_v2/coco_labels.txt

# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
import argparse
import imutils
import time
import cv2
import datetime
import time, json, socket
import collections
from imutils import keyclipwriter
import configparser
import os
buf_len = 50
prob_fac =  0.7
prediction = 'na'
consecFrames = 0 
frames_record = 64
kcw = keyclipwriter.KeyClipWriter(frames_record) 
buffer_prediction = collections.deque(maxlen=buf_len)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to TensorFlow Lite object detection model")
ap.add_argument("-l", "--labels", required=True,
	help="path to labels file")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
folder = '/media/nfs/nfsjetson/'
file_config = 'config.txt'
nfs_loc = folder+"media/person/"
dir_loc = "/home/pi/media/"
# initialize the labels dictionary
print("[INFO] parsing class labels...")
labels = {}

config = configparser.ConfigParser()
config.readfp(open(folder+file_config))
url_path = config.get('camera_one', 'url')
host_udp_server = config.get('camera_one', 'server')



def send_msg_udp(value):
	UDP_IP_ADDRESS = host_udp_server
	UDP_PORT_NO = 6789
	HEADERSIZE = 10
	trans_msg = value
	msg = {}
	msg['message'] = "time: "+str(int(time.time()))+' ,person_detected: '+value
	msg['topic'] =  'aix:vision'
	msg['appKey'] =  'AGlqbaCrEF3UBiMXbThxN4qXNFFHAmeL'
	msg['time'] = int(time.time())
	json_data = json.dumps(msg).encode('utf-8')



	clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	clientSock.sendto(json_data, (UDP_IP_ADDRESS, UDP_PORT_NO))
	print('Sent msg to socket_io')
	# pass



# loop over the class labels file
for row in open(args["labels"]):
	# unpack the row and update the labels dictionary
	(classID, label) = row.strip().split(maxsplit=1)
	labels[int(classID)] = label.strip()

# load the Google Coral object detection model
print("[INFO] loading Coral model...")
model = DetectionEngine(args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
url = url_path#'http://10.0.0.220:8080/?action=stream'
vs = VideoStream(src=url).start()

def check_buf(b):
    # pass
    if b.count(b[0]) > buf_len * prob_fac:
        return b[0]



cap = cv2.VideoCapture(url) # Then start the webcam
cap.set(cv2.CAP_PROP_FPS, 20)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 100)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)



#vs = VideoStream(usePiCamera=False).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	updateConsecFrames = True 
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 500 pixels
	# frame = vs.read()
	ret, frame = cap.read()
	# print(ret)
	if ret:
		frame = imutils.resize(frame, width=500)
		orig = frame.copy()

		# prepare the frame for object detection by converting (1) it
		# from BGR to RGB channel ordering and then (2) from a NumPy
		# array to PIL image format
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = Image.fromarray(frame)

		# make predictions on the input frame
		start = time.time()
		results = model.detect_with_image(frame, threshold=args["confidence"],
			keep_aspect_ratio=True, relative_coord=False)
		end = time.time()
		if(len(results) == 0):
			buffer_prediction.append("na")
			tmp_pred = check_buf(buffer_prediction)
			prediction = "na"
			# count_pred = tmp_pred
			# print('sahoo ', tmp_pred)
			# count_pred = "na"
			# tmp_pred = "na"
			
		# loop over the results
		for r in results:
			
	
			# extract the bounding box and box and predicted class label
			box = r.bounding_box.flatten().astype("int")
			(startX, startY, endX, endY) = box
			label = labels[r.label_id]
			if label is not 'person':
				pass
				# if label is None:
				# 	label = "No Object"
				# buffer_prediction.append(label)
				# cv2.rectangle(orig, (startX, startY), (endX, endY),
				# 		(0, 255, 0), 2)
				# y = startY - 15 if startY - 15 > 15 else startY + 15
				# text = "{}: {:.2f}%".format(label, r.score * 100)
				# cv2.putText(orig, text, (startX, y),
				# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			if label == 'person':
				cv2.rectangle(orig, (startX, startY), (endX, endY),
						(0, 255, 0), 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				text = "{}: {:.2f}%".format(label, r.score * 100)
				cv2.putText(orig, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


			buffer_prediction.append(label)
			tmp_pred = check_buf(buffer_prediction)
			count_pred = tmp_pred
			# print('sahoo2',tmp_pred)
			if (count_pred is not prediction
				and count_pred is not None
				and count_pred == "person"
			):
				print('yay ', count_pred, prediction)
			# if(label == 'person'):
				prediction = count_pred
				
				consecFrames = 0
				msg = 'person' + "_" + str(int(time.time()))
				print("uploaded with prediction", count_pred)

				
				if not kcw.recording:
					timestamp = datetime.datetime.now()
					# p = "{}_{}.avi".format(count_pred,timestamp.strftime("%Y%m%d-%H%M%S"))
					# print(p)
					global p
					p = count_pred + "_" + str(int(time.time())) + ".avi"

					# zebraBlob = bucket.blob('package_delivery/video/'+p)
					# zebraBlob.upload_from_filename(filename=dir_loc+p)
					print(dir_loc + " " + p)
					send_msg_udp(p)
					kcw.start(dir_loc + p, cv2.VideoWriter_fourcc(*"MJPG"), 15)
                # print("recording started")
			if updateConsecFrames:
				consecFrames += 1

                # update the key frame clip buffer
			kcw.update(orig)

			if kcw.recording and consecFrames == frames_record:
				for i in range(50):
					kcw.update(orig)
				kcw.finish()
				try:
					os.system("cp "+dir_loc + p+" "+nfs_loc + p)
				except Exception as e:
					print("Could not copy to nfs location")

				# video_fs_loc = dir_loc + p


				#push to the collection and after 100 continous +ve reads


				# send the log that the person is detected

				# record the video 

				

				# print('person detected')

				# draw the bounding box and label on the image
				cv2.rectangle(orig, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				text = "{}: {:.2f}%".format(label, r.score * 100)
				cv2.putText(orig, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# show the output frame and wait for a key press
		cv2.imshow("Frame", orig)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
