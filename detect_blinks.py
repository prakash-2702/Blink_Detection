# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video NAME OF VIDEO FILE WITH EXTENSION
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	EAR = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return EAR
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,help="path to facial landmark predictor")
'''
if you want to test output with input video file
ap.add_argument("-v", "--video", type=str, default="",help="path to input video file")
'''
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
counter = 0
total = 0

# initialize dlib's face detector (HOG-based) and then createthe facial landmark predictor
print("[Alert] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[Alert] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:

	# grab the frame from the threaded video file stream, resize it, and convert it to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rectangles = detector(gray, 0)

	# loop over the face detections
	for rect in rectangles:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		EAR = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
		if EAR < EYE_AR_THRESH:
			counter += 1

		# otherwise, the eye aspect ratio is not below the blink threshold
		else:
			# if the eyes were closed for a sufficient number of then increment the total number of blinks
			if counter >= EYE_AR_CONSEC_FRAMES:
				total += 1

			# reset the eye frame counter
			counter = 0

		# show the total number of blinks on the frame along with the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `a` key is pressed, break from the loop
	if key == ord("a"):
		break

cv2.destroyAllWindows()
vs.stop()