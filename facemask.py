import cv2
from scipy.spatial import distance as dist
import numpy as np
import dlib
import time
import os

def create_norm(mask):
	""" Takes in the applicable mask, and creates a pixel filter for it. 
	The background pixels, assumed to be white (255,255,255) are converted to ones.
	The pixels inside the mask are converted to zeros.
	"""
	gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	norm = np.zeros_like(mask)
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if gray[i][j] == 255:
				norm[i][j] = np.ones(3)
			else:
				norm[i][j] = np.zeros(3)
	return norm


def shape_to_np(shape):
	""" Takes in an object of class dlib.full_object_detection, iterates through all
	the landmarks, and returns a list of x,y cordinates indexed to their respective
	facial landmark.
	"""
	coords = np.zeros((shape.num_parts, 2), dtype = 'int')

	for i in range(shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords


def eye_aspect_ratio(eye):
	""" Computes the eye aspect ratio from landmark coordinates, using the formula,
	ear = (vertical_1+vertical_2)/(2*horizontal)

	The ratio drops closer to zero when a person blinks.
	"""
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)

	return ear

def weighted_smoothing(face,prev,smooth_factor):
	""" Applies weighted smoothing to the face using the formula S(t) = alpha*S(t)+(1-alpha)*S(t-1).

	I noticed haarcascades have a very jittery face detection, which made the applied facemask pretty 
	unstable. So I decided to use smoothing as a fix, and it works pretty well. For the sake of 
	empirical comparison, change the smooth factor to 1 to eliminate smoothing.

	The function also converts integer position values into float, so they need to be converted back.
	astype('int32') performs a floor operation. Since I want a round operation instead, I use
	this fact to implement round(y) = floor(y+0.5), hence the addition of the 0.5 at the end.
	"""
	face = face*smooth_factor+prev*(1-smooth_factor)
	return (face+0.5).astype('int32')


haar_classifier = 'haarcascade_frontalface_alt.xml'
shape_pred = 'shape_predictor_68_face_landmarks.dat'

clf = cv2.CascadeClassifier(haar_classifier)
predictor = dlib.shape_predictor(shape_pred)

mask_directory = 'masks//'
masks = os.listdir(mask_directory)
num_masks = len(masks)
m = 0

COUNTER = 0
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 10

(lStart, lEnd) = (42,48)
(rStart, rEnd) = (36,42)

mask = cv2.imread(mask_directory+masks[m])
norm = create_norm(mask)

cam_port = 0

print("Starting video...")

vs = cv2.VideoCapture(cam_port,cv2.CAP_DSHOW)
time.sleep(1.0)

trigger = True
prev = np.array([0,0,0,0])
smooth_factor = .05

while True:

	ret,img = vs.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = clf.detectMultiScale(gray, 1.3, 5)
	for face in faces:

		#Smoothing the shaky face position
		if trigger:
			prev = face
			trigger=False
		else:
			face = weighted_smoothing(face,prev,smooth_factor)
			
		(x,y,w,h) = face

		rect = dlib.rectangle(np.asscalar(x),np.asscalar(y),np.asscalar(x+w),np.asscalar(y+w))
		shape = shape_to_np(predictor(gray, rect))

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]

		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		ear = (leftEAR + rightEAR) / 2.0

		print(ear)

		if ear < EYE_AR_THRESH:
			COUNTER += 1
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				m = (m+1)%num_masks
				mask = cv2.imread(mask_directory+masks[m])
				norm = create_norm(mask)
				COUNTER = 0
		else:
			COUNTER = 0


		resized_mask = cv2.resize(mask, (w,h) , interpolation = cv2.INTER_AREA)
		resized_norm = cv2.resize(norm, (w,h) , interpolation = cv2.INTER_AREA)

		cropped_face = np.multiply(img[y:y+h, x:x+w],resized_norm)+resized_mask
		img[y:y+h, x:x+w] = cropped_face

		# For test purposes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

		prev = face

	# flip_img = cv2.flip(img,1)
	
	cv2.imshow('le mask',img)
	
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

#cv2.imwrite('out.png',img)
vs.release()
cv2.destroyAllWindows()
