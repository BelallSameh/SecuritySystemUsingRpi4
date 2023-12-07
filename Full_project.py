#Voice output

import pyttsx3 as py

################################


#Servo setup

import RPi.GPIO as GPIO
from gpiozero import AngularServo
GPIO.setmode(GPIO.BCM)
################################

#LCD setup

from RPLCD import CharLCD# Importing Adafruit library for LCD
from time import sleep # Importing sleep from time library to add delay in program
# initiate lcd and specify pins
lcd = CharLCD(cols=16, rows=2, pin_rs=26, pin_e=19, pins_data=[13, 6, 5, 21]
              , numbering_mode=GPIO.BCM)
lcd.clear()

################################

#Facemask setup

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

################################

#Temperature setup

from smbus2 import SMBus
#from mlx90614 import MLX90614

################################

#Facemask function to be called

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

#############################################

#3-authentication level test

GPIO.setup(18, GPIO.OUT)
servo=AngularServo(18,min_pulse_width=0.0006,max_pulse_width=0.0023)
GPIO.setup(23, GPIO.IN)

while(1):

    lcd.clear()
    bus = SMBus(1)
    sensor = MLX90614(bus, address=0x5A)
    engine = py.init()
    engine.say("Welcome to A A S T. C A I")
    lcd.write_string(u"Welcome to AAST CAI")
    engine.runAndWait()
    sleep(5)
    lcd.clear()
    fingerflag=0
    maskflag=0
    tempflag=0
    #############################################
    engine.say("Scan your finger")
    engine.runAndWait()
    lcd.write_string(u"Scan your finger")
    sleep(7)
    fingerflag=GPIO.input(23)
    sleep(3)
    if(fingerflag==1):
        lcd.clear()
        engine.say("Finger print detected")
        engine.runAndWait()
        lcd.write_string(u"Finger print detected")
        sleep(5)
    elif(fingerflag==0):
        lcd.clear()
        engine.say("Finger print not detected")
        engine.runAndWait()
        lcd.write_string(u"Finger print not detected")
        sleep(5)
        lcd.clear()
        engine.say("Not allowed to enter")
        engine.runAndWait()
        lcd.clear()
        lcd.write_string(u"Not allowed to enter")
        sleep(5)
        lcd.clear()
        continue
    #############################################
    engine.say("Please wait")
    engine.runAndWait()
    lcd.clear()
    lcd.write_string(u"Loading ...")
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args["model"])

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    #vs = VideoStream(src=0).start()
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    lcd.clear()
    engine.say("Face the camera")
    engine.runAndWait()
    lcd.write_string(u"Face the camera")
    sleep(5)

    # grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # detect faces in the frame and determine if they are wearing a
	# face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
	    # the bounding box and text
        if mask > withoutMask:
          lcd.clear()
          maskflag=1
          engine.say("Thanks for wearing mask")
          engine.runAndWait()
          label=("Thanks for wearing mask")
          color = (0, 255, 0)
          lcd.write_string(u"Mask detected")
          sleep(5)
        else:
            lcd.clear()
            maskflag=0
            engine.say("Not mask detected")
            engine.runAndWait()
            lcd.write_string(u"No mask detected Try again")
            sleep(5)
            lcd.clear()
            engine.say("Not allowed to enter")
            engine.runAndWait()
            label=("No mask detected")
            color = (0, 0, 255)
            lcd.write_string(u"Not allowed to enter")
            sleep(5)

    cv2.destroyAllWindows()
    vs.stop()

    if(maskflag==0):
        continue
    #############################################
    lcd.clear()
    lcd.write_string(u"Loading ...")
    sleep(3)
    lcd.clear()
    engine.say("Face the temperature sensor")
    engine.runAndWait()
    lcd.write_string(u"Face the temperature sensor")
    sleep(5)

    y = sensor.get_object_1()
    #y=36
    if(y<37.3):
        lcd.clear()
        engine.say("Temperature is normal")
        engine.runAndWait()
        lcd.write_string(u"Temperature is normal")
        sleep(5)
    else:
        lcd.clear()
        engine.say("Temperature is high")
        engine.runAndWait()
        lcd.write_string(u"Temperature is high")
        sleep(5)
        lcd.clear()
        engine.say("Not allowed to enter")
        engine.runAndWait()
        lcd.write_string(u"Not allowed to enter")
        sleep(5)
        continue
   # bus.close()
    lcd.clear()
    engine.say("Allowed to enter. Welcome to College of artificial intelligence")
    engine.runAndWait()
    lcd.write_string(u"Welcome to C A I")
    servo.angle=90
    sleep(5)
    servo.angle=0
    sleep(5)
    
pwm.stop()
GPIO.cleanup()