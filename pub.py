#pub.py is meant to run on the RaspberryPi. It allows you to wire a button, two LEDs
#and a camera to the Pi. When the button is pressed, it takes a picture with the camera
#and sends it to a MQTT topic (which hopefully the facial recognition server is subscribed to.
#It then awaits a response and displays that response with the LEDs.
#Date: 5/7/2018
#Authors: Brent Ritzema and Toussaint Cruise

import paho.mqtt.client as mqtt
import time
from picamera import PiCamera
import RPi.GPIO as GPIO


#Constants
BROKER = "iot.cs.calvin.edu"
PORT = 1883
QOS = 1
IMAGE_TOPIC = "tcc3/facenet/image"
RESPONSE_TOPIC = "tcc3/facenet/response"
IMAGE_FILENAME = "face.jpg"
BUTTON = 16
MATCH_LED = 21
NOMATCH_LED = 20

#Global Variables
camera = PiCamera()
client = mqtt.Client("P1")

#setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(MATCH_LED, GPIO.OUT)
GPIO.setup(NOMATCH_LED, GPIO.OUT)
GPIO.setup(BUTTON,GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

client.connect(BROKER, PORT, 60)                 # Connect to server
client.loop_start()                              # initial start before loop

def on_connect(client, userdata, rc, *extra_params):
	print('Connected with result code='+str(rc))

#Checks response, if it's a 1 (a match), it lights up the green LED, if it's a 0 (no match)
# it lights up the red LED.
def on_message(mosq, obj, msg):
	print("Response recieved")
	answer = int(msg.payload)
	if answer ==0:
		GPIO.output(NOMATCH_LED, True)
		time.sleep(1)
		GPIO.output(NOMATCH_LED, False)
	else:
		GPIO.output(MATCH_LED, True)
		time.sleep(1)
		GPIO.output(MATCH_LED, False)

#This is a callback for when the button is pressed. It captures an image and
#pushes it to the facial recognition server.
def take_photo():
	camera.start_preview()
	camera.capture(IMAGE_FILENAME)
	camera.stop_preview()
	file = open(IMAGE_FILENAME, "rb")         # open the file, note r = read, b = binary
	imagestring = file.read()                                            # read the file
	byteArray = bytes(imagestring)                                       # convert to byte string
	client.publish(topic= IMAGE_TOPIC, payload= byteArray ,qos= QOS)  # publish it to the MQ queue


#Callbacks
client.on_connect = on_connect
client.on_message = on_message
client.subscribe(RESPONSE_TOPIC, QOS)
GPIO.add_event_detect(BUTTON, GPIO.RISING, callback=take_photo, bouncetime=500)

#wait for camera to 'wake up'
time.sleep(2)

#wait for callbacks
while True:
	pass

camera.close()
GPIO.cleanup()
