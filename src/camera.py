import asyncio
import os
import sys
import aiohttp
import json
import subprocess
import inspect
import ssl
import logging
import datetime
import pathlib
import pickle
import subprocess
import objgraph
import gc
import argparse
import psutil


from aiohttp import web

import cv2
import time
import numpy as np
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FileVideoStream
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils.object_detection import non_max_suppression
from imutils import paths
from imutils.video import FPS
import base64
from base64 import b64encode
from skimage.measure import compare_ssim
import socket
import dlib

from cli import Recon
from draw import Draw


#os.environ['PYTHONASYNCIODEBUG'] = '1'
#ffmpeg -f avfoundation -framerate 30 -i "0" -f mjpeg -q:v 100 -s 640x480 http://localhost:5000/feed
#ffmpeg -re -i ../../sabnzbd/complete/shows/The.Big.Bang.Theory.S11E22.mkv -f mjpeg -q:v 20 -s 640x480 http://localhost:5000/feed
#ffmpeg -f avfoundation -framerate 30 -i "0" -f mpegts udp://192.168.1.26:9999
#rtsp://admin:admin123@190.218.5.228:554

#China:
#rtsp://DNA:DNA2017!@98.186.153.94:554

#Panam
#rtsp://admin:Admin1234@190.141.212.109:554/Streaming/channels/1602
#rtsp://admin:Admin1234@181.197.134.254:554/Streaming/channels/1602

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--local", action='store_true', required=False,
	help="run outside docker")
ap.add_argument("-v", "--video", action='store_true', required=False, default=False,
	help="read from video")
args = vars(ap.parse_args())

#LOGS
logging.basicConfig(format="[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s")

logger = logging.getLogger('async')
logger.setLevel(logging.DEBUG)

fc = {}
local = {}

proc = psutil.Process(os.getpid())
logger.debug("mem: {}".format(proc.get_memory_info().rss))
objgraph.show_most_common_types()

def init():
	global fc

	local['draw'] = Draw()

	fc['ip'] = socket.gethostbyname(socket.gethostname())
	fc['hostname'] = socket.gethostname()

	#reload object from file
	try:
		file2 = open(r'./.config', 'rb')
		fc = pickle.load(file2)
		file2.close()
		local['feedURL'] = fc['feedURL']
		if local['feedURL'].startswith('rtsp'):
			args["video"] = False
		else:
			args["video"] = True
	except:
		#Face Recognition parameters
		fc['tolerance'] = 0.5
		fc['show_distance'] = False
		fc['prototxt'] = "deploy.prototxt.txt"
		fc['prototxt_object'] = "MobileNetSSD_deploy.prototxt.txt"
		fc['model'] = "res10_300x300_ssd_iter_140000.caffemodel"
		fc['model_object'] = "MobileNetSSD_deploy.caffemodel"
		fc['shape_predictor'] = "shape_predictor_68_face_landmarks.dat"
		fc['confidence'] = 0.25
		fc['confidence_obj'] = 0.8
		if args["video"]:
			if args["local"]:
				local['feedURL'] = "/Users/admin/Downloads/abc3.mp4"
				#local['feedURL'] = 0
			else:
				local['feedURL'] = "/tmp/abc3.mp4"
		else:
			local['feedURL'] = "rtsp://admin:admin123@190.218.236.232:555/cam/realmonitor?channel=15&subtype=0"
			#local['feedURL'] = "rtsp://admin:admin123@192.168.0.107:554/cam/realmonitor?channel=15&subtype=0"
			#local['feedURL'] = "rtsp://admin:admin@192.168.0.61:554/"

		fc['camera_name'] = fc['hostname']
		fc['enableObjectDetection'] = True


	# Load an color image
	local['emptyFeed'] = cv2.imread('nofeed.jpg',1)

	proc = subprocess.Popen(["curl", "--unix-socket", "/var/run/docker.sock", "http://localhost/containers/"+fc['hostname']+"/json"], stdout=subprocess.PIPE)
	#proc = subprocess.Popen(["curl", "--unix-socket", "/var/run/docker.sock", "http://localhost/containers/f2ab98cd4eb0/json"], stdout=subprocess.PIPE)
	stdout, stderr = proc.communicate()

	j = json.loads(stdout)
	try:
		array = j['NetworkSettings']['Ports']['5000/tcp']
		fc['port'] = array[0]['HostPort']
	except KeyError:
		fc['port'] = 5000

	logger.debug(fc)

	# initialize the list of class labels MobileNet SSD was trained to
	# detect, then generate a set of bounding box colors for each class
	local['classes'] = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person"]
	local['colors'] = np.random.uniform(0, 255, size=(len(local['classes']), 3))


	local['capture'] = {}
	local['ws_msg'] = {}
	local['shutdown'] = False
	local['gimage'] = None
	local['known_people_folder'] = 'known/camera'
	local['timer'] = 5

	local['modulo'] = 1
	local['color_unknow'] = (0, 0, 255)
	local['color_know'] = (162, 255, 0)
	local['thickness'] = 2
	local['number_fps_second'] = 1/30
	local['quality'] = 100
	local['banner'] = "hello..."
	local['color_banner'] = (0, 0, 255)

	local['video_size'] = 800
	local['control'] = None
	local['ffmpeg_feed'] = False

	# load our serialized model from disk
	logger.info("loading models...")
	local['net'] = cv2.dnn.readNetFromCaffe(fc['prototxt'], fc['model'])
	local['net_object'] = cv2.dnn.readNetFromCaffe(fc['prototxt_object'], fc['model_object'])

	start_encoding = time.time()
	logger.info("loading knowns...")
	local['face_recognition'] = Recon(local['known_people_folder'], fc['tolerance'], fc['show_distance'])
	logger.info('encoding of the known folder completed in: {:.2f}s'.format(time.time() - start_encoding))

	# initialize the HOG descriptor/person detector
	local['hog'] = cv2.HOGDescriptor()
	local['hog'].setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor and the face aligner
	local['detector'] = dlib.get_frontal_face_detector()
	local['predictor'] = dlib.shape_predictor(fc['shape_predictor'])
	local['fa'] = FaceAligner(local['predictor'], desiredFaceWidth=256)

	#Check if OpenCV is Optimized:
	logger.info("CV2 Optimized: {}".format(cv2.useOptimized()))

async def add(_id, photo, firstname, lastname, _oldName):
	newName = _id + '.' + firstname + '_' + lastname
	oldName = '0.' + _oldName
	try:
		with open("./known/camera/"+_id+"."+ firstname+"_"+ lastname +".jpg", "wb") as fh:
			fh.write(base64.decodebytes(str.encode(photo)))
	except:
		pass
	await local['face_recognition'].delete_unknown_names(oldName)
	await local['face_recognition'].unknown_people( None, newName, cv2.imdecode(np.frombuffer(base64.decodebytes(str.encode(photo)), dtype=np.uint8), cv2.IMREAD_COLOR))
	try:
		local['capture'][newName] = local['capture'][oldName]
		del local['capture'][oldName]
		logger.info("Added")
	except Exception as e:
		logger.info ("Unexpected error: ", e)
		pass

#CHANGE KNOWN FOLDER TO: VAR
async def delete(_id, firstname, lastname):
	try:
		os.remove("./known/camera/" + _id + '.' + firstname + '_' + lastname + ".jpg")
	except:
		pass
	del local['capture'][_id + '.' + firstname + '_' + lastname]
	await local['face_recognition'].delete_unknown_names(_id + '.' + firstname + '_' + lastname)
	logger.info ('delete{}'.format(_id))

async def read_frame(image):
	try:
		local['frames'] = local['frames'] + 1

		try:
			orig = image.copy()
		except:
			image = local['emptyFeed']
			orig = image.copy()

		display = imutils.resize(orig, width=local['video_size'])

		if (local['frames'] % local['modulo'] == 0):

			# load the image and resize it to (1) reduce detection time
			# and (2) improve detection accuracy

			rp = 400.0 / image.shape[1]
			dimp = (400, int(image.shape[0] * rp))

			image = cv2.resize(image, dimp, interpolation = cv2.INTER_AREA)

			# detect people in the image
			(rects, weights) = local['hog'].detectMultiScale(image, winStride=(4, 4),
				padding=(8, 8), scale=1.05)

			# apply non-maxima suppression to the bounding boxes using a
			# fairly large overlap threshold to try to maintain overlapping
			# boxes that are still people
			rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
			pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

			# draw the final bounding boxes
			minX = 100000
			minY = 100000
			maxW = 0
			maxH = 0

			for (xA, yA, xB, yB) in pick:
				if xA < minX: minX = xA
				if yA < minY: minY = yA
				if xB > maxW: maxW = xB
				if yB > maxH: maxH = yB

			cropped = image[minY:maxH, minX:maxW]

			if maxH + maxW != 0:

				### BRILLO ###
				b = 64. # brightness
				c = 0.  # contrast

				#call addWeighted function, which performs:
				#    dst = src1*alpha + src2*beta + gamma
				# we use beta = 0 to effectively only operate on src1
				face = cv2.addWeighted(cropped, 1. + c/127., cropped, 0, b-c)
				orig = cv2.addWeighted(orig, 1. + c/127., orig, 0, b-c)

				### FACE RECOGNITION
				(h, w) = face.shape[:2]

				blob = cv2.dnn.blobFromImage(cv2.resize(face, (300, 300)), 1.0,
								(300, 300), (104.0, 177.0, 123.0))

				# pass the blob through the network and obtain the detections and
				# predictions
				local['net'].setInput(blob)
				detections = local['net'].forward()

				box = {}
				box_obj = {}
				text = {}
				result = {}

				for i in range(0, detections.shape[2]):

					# extract the confidence (i.e., probability) associated with the
					# prediction
					confidence = detections[0, 0, i, 2]

					# filter out weak detections by ensuring the `confidence` is
					# greater than the minimum confidence

					if confidence > fc['confidence']:
						#logger.debug('confidence: {} / {}'.format(confidence,fc['confidence']))

						# compute the (x, y)-coordinates of the bounding box for the
						# object
						box[i] = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box[i].astype("int")

						# crop the face(s)
						face = face[startY:endY, startX:endX]

						if startY < 1 or endY < 1 or startX < 1 or endX < 1:
							continue

						if startY > 300 or endY > 300 or startX > 300 or endX > 300:
							continue

						ratioY = int(minY*3.2)
						ratioH = int(maxH*3.2)
						ratioX = int(minX*3.2)
						ratioW = int(maxW*3.2)

						#print("[INFO] Y={} H={} X={} W={}".format(minY, maxH, minX, maxW))

						startY = int(startY*3.2)
						endY = int(endY*1.3)
						startX = int(startX*3.2)
						endX = int(endX*1.3)

						#print("[INFO] Y'={} H'={} X'={} W'={}".format(startY, endY, startX, endX))

						ratioY = startY + ratioY
						ratioH = endY + ratioY
						ratioX = startX + ratioX
						ratioW = endX + ratioX

						#cv2.imshow("FACE", face)
						txt = "{:.2f}%".format(confidence * 100)
						text[i] = txt

						cropped = orig[ratioY:ratioH, ratioX:ratioW]
						gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
						rects = local['detector'](gray, 2)

						for rect in rects:
							#logger.debug("in the rects")
							# extract the ROI of the *original* face, then align the face using facial landmarks
							(x, y, w, h) = rect_to_bb(rect)
							faceAligned = local['fa'].align(cropped, gray, rect)

							array = []
							distances = []
							array, distances = await local['face_recognition'].test_image(faceAligned)
							#array = ['0.unknown']
							#logger.debug("array: {}".format(array))
							#logger.debug("distances: {}".format(distances))
							result[i] = 'unknown'
							logger.debug("capture #: {}".format(len(local['capture'])))

							if (array != ['0.unknown']) and (array != []):
								try:
									local['capture'][array[0]]
								except (KeyError, IndexError) as e:
									local['capture'][array[0]] = [faceAligned, box[i], text[i]]

									try:
										retval, jpg = cv2.imencode('.jpg', faceAligned)
										base64_bytes = b64encode(jpg)
										wsMessage = json.dumps({"action":"add_known", "date": time.strftime("%Y%m%d%H%M%S"), "name": str.split(array[0], '.')[0], "confidence":text[i], "image": base64_bytes.decode('utf-8'), "camera":fc['camera_name']})

										#logger.info('add known: {}'.format(str.split(array[0], '.')[0]))
										await local['control'].send_str(wsMessage)
									except Exception as ex:
											local['ws_msg'] = wsMessage
											exc_type, exc_obj, exc_tb = sys.exc_info()
											template = "an exception of type {0} occurred. arguments:\n{1!r}"
											message = template.format(type(ex).__name__, ex.args)
											logger.error("error sending message line: {}".format(exc_tb.tb_lineno))
											logger.error(message)

								if array[0].startswith( '0.unknown' ):
									color = local['color_unknow']
								else:
									color = local['color_know']
								result[i] = str.split(array[0], '.')[1]
							else:
								color = local['color_unknow']
								add = await local['face_recognition'].unknown_people(faceAligned, '0.unknown'+str(local['unknown']), None)
								logger.debug("unknow: {}".format(add))

								if add != 0:
									try:
										retval, jpg = cv2.imencode('.jpg', faceAligned)
										base64_bytes = b64encode(jpg)
										wsMessage = json.dumps({"action":"add_unknown", "date": time.strftime("%Y%m%d%H%M%S"), "name": 'unknown'+str(local['unknown']), "confidence":text[i], "image": base64_bytes.decode('utf-8'), "camera":fc['camera_name']})
										logger.info('add unknown: {}'.format('unknown'+str(local['unknown'])))
										await local['control'].send_str(wsMessage)

										local['capture']['0.unknown'+str(local['unknown'])] = [faceAligned, box[i], text[i]]

									except Exception as ex:
										local['ws_msg'] = wsMessage
										exc_type, exc_obj, exc_tb = sys.exc_info()
										template = "an exception of type {0} occurred. arguments:\n{1!r}"
										message = template.format(type(ex).__name__, ex.args)
										logger.error("error sending message line: {}".format(exc_tb.tb_lineno))
										logger.error(message)

										result[i] = 'unknown'+str(local['unknown'])
							local['unknown'] = local['unknown'] + 1


							if array[0].startswith( '0.unknown' ):
								color = local['color_unknow']
							else:
								color = local['color_know']

							#overlay = local['draw'].face(display, 10, 10, ratioX, ratioY, ratioH, ratioW, (endX-((endX-startX)-faceEndX)), startY, (endY-((endY-startY)-faceEndY)), color, local['thickness'], text[i], result[i])

				del local['box']
				del local['text']
				del local['result']
				local['box'] = {}
				local['text'] = {}
				local['result'] = {}

				local['box'] = box
				local['text'] = text
				local['result'] = result

		else:
			if len(local['box']) > 0:

				for i in range(0, len(local['box'])):
					(startX, startY, endX, endY) = local['box'][i].astype("int")
					(x1, y1, x2, y2) = local['box'][i].astype("int")
					y = startY - 10 if startY - 10 > 10 else startY + 10

					# if local['result'][i].startswith( 'unknown' ):
					# 	color = local['color_unknow']
					# else:
					# 	color = local['color_know']

					#overlay = local['draw'].face(overlay, 10, 10, startX, y, endY, x1, x2, y1, y2, color, local['thickness'], local['text'][i], local['result'][i])

	except Exception as ex:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		template = "an exception of type {0} occurred. arguments:\n{1!r}"
		message = template.format(type(ex).__name__, ex.args)
		logger.error("error sending message line: {}".format(exc_tb.tb_lineno))
		logger.error(message)
	return display

async def process_video():

	local['frames'] = 0
	local['unknown'] = 0
	local['box'] = {}
	local['text'] = {}
	local['result'] = {}

	try:
		if local['feedURL'].startswith('rtsp'):
			local['_camera'] = WebcamVideoStream(src=local['feedURL']).start()
			#local['_camera'] = VideoStream(local['feedURL']).start()
		else:
			local['_camera'] = cv2.VideoCapture(local['feedURL'])
		await asyncio.sleep(2)
	except Exception as ex:
		template = "an exception of type {0} occurred. arguments:\n{1!r}"
		message = template.format(type(ex).__name__, ex.args)
		logger.error("error processing video:")
		logger.error(message)

	start_timer = time.time()
	while not local['shutdown']:
		end_timer = time.time()
		if (end_timer - start_timer) > local['timer']:
			logger.debug("memory:")
			objgraph.show_most_common_types()
			logger.debug("growth:")
			objgraph.show_growth()
			gc.collect()
			if len(local['capture']) == 0:
				del local['capture']
				local['capture'] = {}


			if local['capture']:
				el = min(local['capture'])
				error = local['capture'].pop(el, None)
				if error is not None:
					if el.startswith( '0.unknown' ):
						await local['face_recognition'].delete_unknown_names(el)
			start_timer = time.time()

		c_fps = FPS().start()
		if local['feedURL'].startswith('rtsp'):
			image = local['_camera'].read()
		else:
			try:
				grab, image = local['_camera'].read()
			except:
				print ("grab: ", grab)
				image = local['emptyFeed']

		image = await read_frame(image)

		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), local['quality']]
		rslt, image = cv2.imencode('.jpg', image, encode_param)
		local['gimage'] = image
		local['wsimage'] = image

		c_fps.update()
		c_fps.stop()
		#print (c_fps.elapsed())
		#print (c_fps.fps())

		await asyncio.sleep(local['number_fps_second'])

async def remoteFeedHandle():
	logger.info("remote feed handle")
	while not local['shutdown']:
		try:
			await ws.send_bytes(local['wsimage'].tostring())
		except Exception as ex:
			template = "an exception of type {0} occurred. arguments:\n{1!r}"
			message = template.format(type(ex).__name__, ex.args)
			logger.error("error connecting to control center:")
			logger.error(message)
			await asyncio.sleep(1)
		await asyncio.sleep(local['number_fps_second'])

async def baseHandle(request):
	logger.info ('feed local handle')
	return web.Response(text="Server "+ fc['hostname'] +" is Up!!!")

async def localFeedHandle(request):
	logger.info ('local feed handle')
	response = web.StreamResponse()
	response.content_type = ('multipart/x-mixed-replace; '
													 'boundary=--frameboundary')
	await response.prepare(request)

	async def write(img_bytes):
		"""Write image to stream."""
		await response.write(bytes(
				'--frameboundary\r\n'
				'Content-Type: {}\r\n'
				'Content-Length: {}\r\n\r\n'.format(
						response.content_type, len(img_bytes)),
				'utf-8') + img_bytes + b'\r\n')

	while not local['shutdown']:
		try:
			await write(local['gimage'].tostring())
		except Exception as ex:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			template = "an exception of type {0} occurred. arguments:\n{1!r}"
			message = template.format(type(ex).__name__, ex.args)
			logger.error("error sending message line: {}".format(exc_tb.tb_lineno))
			logger.error(message)
			await asyncio.sleep(1)
		await asyncio.sleep(local['number_fps_second'])

async def heartbeat(timer):
	while local['control'] == None:
		logger.debug('waiting...')
		await asyncio.sleep(1)

	while not local['shutdown']:
		logger.info('heartbeat control center')
		if local['control'] != None:
			try:
				await local['control'].send_str(json.dumps({"action":"hello","port":fc['port'],"ip":fc['ip'],"hostname":fc['hostname'],"date":time.strftime("%Y%m%d%H%M%S")}))
			except Exception as ex:
				del local['control']
				local['control'] = None

				template = "an exception of type {0} occurred. arguments:\n{1!r}"
				message = template.format(type(ex).__name__, ex.args)
				logger.error("error connecting to control center:")
				logger.error(message)
		else:
			logger.error('heartbeat lost to control center')

		await asyncio.sleep(timer)

async def control(timer):
	while not local['shutdown']:
		logger.info('connecting to control center')
		try:
			link = f'ws://control:1880/node-red/control'
			if args["local"]:
				link = f'wss://exception34.com/node-red/control'
			async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
				async with session.ws_connect(link, heartbeat=2, receive_timeout=5) as ws:
					del local['control']
					local['control'] = ws
					logger.info ('control connected')
					async for msg in ws:
						payload = json.loads(msg.data)
						logger.info ("message: {}".format(payload['action']))

						if msg.type == aiohttp.WSMsgType.TEXT:
							if payload['action'] == "debug":
								if payload['id'] == fc['hostname']:
									local['toggleDebug'] = not local['toggleDebug']

							if payload['action'] == "object":
								if payload['id'] == fc['hostname']:
									fc['enableObjectDetection'] = not fc['enableObjectDetection']

							elif payload['action'] == "add":
								#logger.debug(payload['id'],payload['firstname'], payload['lastname'],payload['oldName'],payload['treatment'])
								await add(payload['id'],payload['photo'],payload['firstname'], payload['lastname'],payload['oldName'])

							elif payload['action'] == "delete":
								await delete(payload['id'],payload['firstname'], payload['lastname'])

							elif payload['action'] == "start":
								if payload['id'] == fc['hostname']:
									local['feedler'] = asyncio.ensure_future(process_video())

							elif payload['action'] == "stop":
								if payload['id'] == fc['hostname']:
									local['feedler'].cancel()
									await local['feedler']
									await local['feed'].close()
									local['feed_task'].cancel()
									await local['feed_task']

							elif payload['action'] == "quality":
								if payload['id'] == fc['hostname']:
									local['quality'] = int(payload['value'])

							elif payload['action'] == "fps":
								if payload['id'] == fc['hostname']:
									local['number_fps_second'] = float(payload['value'])

							elif payload['action'] == "modulo":
								if payload['id'] == fc['hostname']:
									local['modulo'] = int(payload['value'])

							elif payload['action'] == "mjpeg":
								if payload['id'] == fc['hostname']:
									local['mjpegToggle'] = int(payload['value'])
									if local['mjpegToggle'] == 1:
										local['feed_task'] = asyncio.ensure_future(feedsocket())
										#local['mjpeg_task'] = asyncio.ensure_future(remoteFeedHandle())
									else:
										local['mjpeg_task'].cancel()
										await local['mjpeg_task']
										local['feedler'].cancel()
										await local['feedler']

							elif payload['action'] == "size":
								logger.debug(payload['id'], fc['hostname'])
								if payload['id'] == fc['hostname']:
									local['video_size'] = int(payload['value'])

							elif payload['action'] == "feed":
								logger.info(payload['url'])
								if payload['id'] == fc['hostname']:
									local['camera_name'] = payload['name']
									if payload['url'] == "0":
										local['feedURL'] = 0
									else:
										local['feedURL'] = payload['url']

									fc['feedURL'] = local['feedURL']
									if payload['url'].startswith('rtsp'):
										local['_camera'].stop()
										local['_camera'] = WebcamVideoStream(src=local['feedURL']).start()
									else:
										local['_camera'] = cv2.VideoCapture(local['feedURL'])

							elif msg.type == aiohttp.WSMsgType.CLOSED:
								break
							elif msg.type == aiohttp.WSMsgType.ERROR:
								break
		except Exception as ex:
			template = "an exception of type {0} occurred. arguments:\n{1!r}"
			message = template.format(type(ex).__name__, ex.args)
			logger.error("error connecting to control center:")
			logger.error(message)
			await asyncio.sleep(timer)

async def feedsocket(timer):
	while not local['shutdown']:
		logger.info('connecting to control center')
		try:
			link = f'ws://control:1880/node-red/feed2'
			if args["local"]:
				link = f'wss://exception34.com/node-red/feed2'
			async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
				async with session.ws_connect(link, heartbeat=2, receive_timeout=5) as ws:
					local['feed'] = ws
					logger.info ('feed connected')
					async for msg in ws:
						logger.info (msg)
		except Exception as ex:
			template = "an exception of type {0} occurred. arguments:\n{1!r}"
			message = template.format(type(ex).__name__, ex.args)
			logger.error("error connecting to control center:")
			logger.error(message)
			await asyncio.sleep(timer)

async def on_shutdown(app):
	logger.info('shutdown...')
	pass

async def cleanup_background_tasks(app):
	logger.info('cleanup background tasks...')
	try:
		await local['feed'].close()
	except Exception as ex:
		template = "an exception of type {0} occurred. arguments:\n{1!r}"
		message = template.format(type(ex).__name__, ex.args)
		logger.error("error connecting to control center:")
		logger.error(message)
	finally:
		logger.info ("cleanup background tasks completed.")

async def build_server(loop, address, port):
		app = web.Application(loop=loop)

		app.router.add_route('GET', '/feed', localFeedHandle)
		app.router.add_route('GET', '/', baseHandle)
		#app.on_cleanup.append(cleanup_background_tasks)
		#app.on_shutdown.append(on_shutdown)
		if args["local"]:
			return await loop.create_server(app.make_handler(), address, port)
		else:
			sslcontext = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
			sslcontext.load_cert_chain('./exception34.crt', './exception34.com.key')
			return await loop.create_server(app.make_handler(), address, port, ssl=sslcontext)

if __name__ == '__main__':
		init()
		loop = asyncio.get_event_loop()
		try:
			asyncio.ensure_future(control(10))
			asyncio.ensure_future(heartbeat(20))
			local['feedler'] = asyncio.ensure_future(process_video())

			loop.run_until_complete(build_server(loop, '0.0.0.0', 5000))

			loop.run_forever()
		except KeyboardInterrupt:
			 local['shutdown'] = True
		finally:
			afile = open(r'./.config', 'wb')
			logger.debug(fc)
			pickle.dump(fc, afile)
			afile.close()

			logger.info("Shutting down {} tasks".format(len(asyncio.Task.all_tasks())))
			for task in asyncio.Task.all_tasks():
					logger.debug(f'Cancelling: {task}')
			for task in asyncio.Task.all_tasks():
					task.cancel()
					loop.run_until_complete(task)

			loop.close()
			logger.info("Shutting Down!")
			sys.exit(1)



