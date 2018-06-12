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

from aiohttp import web

import cv2
import time
import numpy as np
import imutils
from imutils.video import WebcamVideoStream
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils.video import FPS
from cli import Recon
import json
import base64
from base64 import b64encode
from skimage.measure import compare_ssim
import socket

print()
#os.environ['PYTHONASYNCIODEBUG'] = '1'

#LOGS
#formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
logging.basicConfig(format="[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s")

# create a file handler
# handler = logging.FileHandler('camera.log')
# handler.setFormatter(formatter)
# handler.setLevel(logging.INFO)

logger = logging.getLogger('async')
#logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
#logger.addHandler(handler)


fc = {}
feed = {}
local = {}
comm = {}


def init():
	global feed
	global fc

	#reload object from file
	try:
		file2 = open(r'./.config', 'rb')
		fc = pickle.load(file2)
		file2.close()
		feed['feedURL'] = fc['feedURL']
	except:
		#Face Recognition parameters
		fc['tolerance'] = 0.68
		fc['show_distance'] = True
		fc['prototxt'] = "deploy.prototxt.txt"
		fc['model'] = "res10_300x300_ssd_iter_140000.caffemodel"
		fc['shape_predictor'] = "shape_predictor_68_face_landmarks.dat"
		fc['confidence'] = 0.8
		feed['feedURL'] = "rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov"
		try:
			fc['port'] = os.environ['HOST_PORT']
		except:
			fc['port'] = '5000'

	fc['ip'] = socket.gethostbyname(socket.gethostname())
	fc['hostname'] = socket.gethostname()

	local['_capture'] = {}
	local['shutdown'] = False
	local['gimage'] = None
	local['known_people_folder'] = 'known'
	local['timer'] = 60

	# logger.info("loading shape-predictor...")
	# fc['detector'] = dlib.get_frontal_face_detector()
	# fc['predictor'] = dlib.shape_predictor(args["shape_predictor"])
	# fc['fa'] = FaceAligner(predictor, desiredFaceWidth=256)

	feed['toggleDebug'] = False
	feed['banner'] = False

	feed['modulo'] = 5
	feed['color_unknow'] = (0, 0, 255)
	feed['color_know'] = (162, 255, 0)
	feed['thickness'] = 2
	feed['number_fps_second'] = 1/25
	feed['quality'] = 5
	feed['banner'] = "hello..."
	feed['color_banner'] = (0, 0, 255)

	feed['video_size'] = 600
	print (fc)

	#ffmpeg -f avfoundation -framerate 30 -i "0" -f mjpeg -q:v 100 -s 640x480 http://localhost:5000/feed
	#ffmpeg -re -i ../../sabnzbd/complete/shows/The.Big.Bang.Theory.S11E22.mkv -f mjpeg -q:v 20 -s 640x480 http://localhost:5000/feed
	#ffmpeg -f avfoundation -framerate 30 -i "0" -f mpegts udp://192.168.1.26:9999
	#TODO: add ffmpeg internally

	feed['ffmpeg_feed'] = False

	# load our serialized model from disk
	logger.info("loading model...")
	local['net'] = cv2.dnn.readNetFromCaffe(fc['prototxt'], fc['model'])

	start_encoding = time.time()
	local['face_recognition'] = Recon(local['known_people_folder'], fc['tolerance'], fc['show_distance'])
	logger.info('encoding of the known folder completed in: {:.2f}s'.format(time.time() - start_encoding))

	#Check if OpenCV is Optimized:
	logger.info("CV2 Optimized: {}".format(cv2.useOptimized()))

async def read_frame(_frames, _last_image, _count_unknown, _box, _text, _result):
	err = 0
	check = 0
	squares = 0
	speed = 0
	score = 0
	_frames = _frames + 1

	c_fps = FPS().start()
	image = feed['_camera'].read()
	overlay = imutils.resize(image, width=feed['video_size'])

	overlay = cv2.flip( overlay, 1 )

	# if len(_last_image) > 0:
	# 	err = np.sum((overlay.astype("float") - _last_image.astype("float")) ** 2)
	# 	err /= float(overlay.shape[0] * overlay.shape[1])
	# 	#print (err)

	if (_frames % feed['modulo'] == 0):

		# Make copies of the frame for transparency processing
		#overlay = frame.copy()
		#output = overlay.copy()

		#set transparency value
		#alpha  = 0.5

		# make semi-transparent bounding box
		#cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

		# grab the frame dimensions and convert it to a blob
		(h, w) = overlay.shape[:2]

		blob = cv2.dnn.blobFromImage(cv2.resize(overlay, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the detections and
		# predictions
		local['net'].setInput(blob)
		detections = local['net'].forward()


		box = {}
		text = {}
		result = {}
		# loop over the detections
		for i in range(0, detections.shape[2]):


			# extract the confidence (i.e., probability) associated with the
			# prediction
			_confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if _confidence < fc['confidence']:
				continue

			squares = squares + 1
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box[i] = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box[i].astype("int")
			(x1, y1, x2, y2) = box[i].astype("int")

			# draw the bounding box of the face along with the associated
			# probability
			txt = "{:.2f}%".format(_confidence * 100)
			text[i] = txt
			y = startY - 10 if startY - 10 > 10 else startY + 10

			#gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
			cropped = overlay[startY:endY, startX:endX]

			start_encoding = time.time()
			found = False
			array = []
			distances = []
			array, distances = await local['face_recognition'].test_image(cropped)
			logger.debug("array: {}".format(array))
			result[i] = 'unknown'
			if (array != ['0.unknown']) and (array != []):
				# If Customer Known
				try:
					local['_capture'][array[0]]
				except (KeyError, IndexError) as e:
					if startY-50 < 0:
							x1 = 0
					else:
							x1 = startY-50
					if endY+50 < 0:
							y1 = 0
					else:
							y1 = endY+50
					if startX-50 < 0:
							x2 = 0
					else:
							x2 = startX-50
					if endX+50 < 0:
							y2 = 0
					else:
							y2 = endX+50
					saved = overlay[x1:y1, x2:y2]

					local['_capture'][array[0]] = [cropped, box[i], text[i]]

					retval, jpg = cv2.imencode('.jpg', saved)
					base64_bytes = b64encode(jpg)

					control = None
					while control == None:
						try:
							control = comm['control']
							logger.info('add known: {}'.format(str.split(array[0], '.')[0]))
							await control.send_str(json.dumps({"action":"add_known", "date": time.strftime("%Y%m%d%H%M%S"), "name": str.split(array[0], '.')[0], "confidence":text[i], "image": base64_bytes.decode('utf-8'), "camera":fc['hostname']}))#, "image":  cropped.tolist()}))
						except TimeoutError:
							logger.info("timeout")
						finally:
							#logger.info("ws error: ", e)
							await asyncio.sleep(0)
				if array[0].startswith( '0.unknown' ):
					color = feed['color_unknow']
				else:
					color = feed['color_know']
				result[i] = str.split(array[0], '.')[1]
			else:
				#If Customer unknown
				color = feed['color_unknow']
				add = await local['face_recognition'].unknown_people(cropped, '0.unknown'+str(_count_unknown), None)
				if add == 1:
					if startY-50 < 0:
							x1 = 0
					else:
							x1 = startY-50
					if endY+50 < 0:
							y1 = 0
					else:
							y1 = endY+50
					if startX-50 < 0:
							x2 = 0
					else:
							x2 = startX-50
					if endX+50 < 0:
							y2 = 0
					else:
							y2 = endX+50
					saved = overlay[x1:y1, x2:y2]
					local['_capture']['0.unknown'+str(_count_unknown)] = [cropped, box[i], text[i]]
					retval, jpg = cv2.imencode('.jpg', saved)
					base64_bytes = b64encode(jpg)
					control = None
					while control == None:
						try:
							control = comm['control']
							logger.info('add unknown: {}'.format('unknown'+str(_count_unknown)))
							await control.send_str(json.dumps({"action":"add_unknown", "date": time.strftime("%Y%m%d%H%M%S"), "name": 'unknown'+str(_count_unknown), "confidence":text[i], "image": base64_bytes.decode('utf-8'), "camera":fc['hostname']}))#, "image":  cropped.tolist()}))
						except Exception as e:
							logger.info("error: ", e)
							await asyncio.sleep(0)

					result[i] = 'unknown'+str(_count_unknown)
				_count_unknown = _count_unknown + 1

			speed = time.time() - start_encoding

			r = 10
			d = 10

			score = 1
			_last_image = overlay
			local['gimage'] = overlay.copy()
			encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), feed['quality']]
			rslt, local['img_str'] = cv2.imencode('.jpg', overlay, encode_param)
			overlay = cv2.imdecode(local['img_str'], 1)

			cv2.line(overlay, (x1 + r, y1), (x1 + r + d, y1), color, feed['thickness'])
			cv2.line(overlay, (x1, y1 + r), (x1, y1 + r + d), color, feed['thickness'])
			cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, feed['thickness'])

			# Top right drawing
			cv2.line(overlay, (x2 - r, y1), (x2 - r - d, y1), color, feed['thickness'])
			cv2.line(overlay, (x2, y1 + r), (x2, y1 + r + d), color, feed['thickness'])
			cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, feed['thickness'])

			# Bottom left drawing
			cv2.line(overlay, (x1 + r, y2), (x1 + r + d, y2), color, feed['thickness'])
			cv2.line(overlay, (x1, y2 - r), (x1, y2 - r - d), color, feed['thickness'])
			cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, feed['thickness'])

			# Bottom right drawing
			cv2.line(overlay, (x2 - r, y2), (x2 - r - d, y2), color, feed['thickness'])
			cv2.line(overlay, (x2, y2 - r), (x2, y2 - r - d), color, feed['thickness'])
			cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, feed['thickness'])

			cv2.putText(overlay, text[i], (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, feed['thickness'])

			cv2.putText(overlay, result[i], (x1 + r, endY),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, feed['thickness'])

			cv2.line(local['gimage'], (x1 + r, y1), (x1 + r + d, y1), color, feed['thickness'])
			cv2.line(local['gimage'], (x1, y1 + r), (x1, y1 + r + d), color, feed['thickness'])
			cv2.ellipse(local['gimage'], (x1 + r, y1 + r), (r, r), 180, 0, 90, color, feed['thickness'])

			# Top right drawing
			cv2.line(local['gimage'], (x2 - r, y1), (x2 - r - d, y1), color, feed['thickness'])
			cv2.line(local['gimage'], (x2, y1 + r), (x2, y1 + r + d), color, feed['thickness'])
			cv2.ellipse(local['gimage'], (x2 - r, y1 + r), (r, r), 270, 0, 90, color, feed['thickness'])

			# Bottom left drawing
			cv2.line(local['gimage'], (x1 + r, y2), (x1 + r + d, y2), color, feed['thickness'])
			cv2.line(local['gimage'], (x1, y2 - r), (x1, y2 - r - d), color, feed['thickness'])
			cv2.ellipse(local['gimage'], (x1 + r, y2 - r), (r, r), 90, 0, 90, color, feed['thickness'])

			# Bottom right drawing
			cv2.line(local['gimage'], (x2 - r, y2), (x2 - r - d, y2), color, feed['thickness'])
			cv2.line(local['gimage'], (x2, y2 - r), (x2, y2 - r - d), color, feed['thickness'])
			cv2.ellipse(local['gimage'], (x2 - r, y2 - r), (r, r), 0, 0, 90, color, feed['thickness'])

			cv2.putText(local['gimage'], text[i], (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, feed['thickness'])

			cv2.putText(local['gimage'], result[i], (x1 + r, endY),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, feed['thickness'])

		_box = box
		_text = text
		_result = result
		_last_image = overlay
		if score != 1:
			local['gimage'] = overlay.copy()
			encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), feed['quality']]
			rslt, local['img_str'] = cv2.imencode('.jpg', overlay, encode_param)
			overlay = cv2.imdecode(local['img_str'], 1)

	else:
		_last_image = overlay
		local['gimage'] = overlay.copy()
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), feed['quality']]
		rslt, local['img_str'] = cv2.imencode('.jpg', overlay, encode_param)
		overlay = cv2.imdecode(local['img_str'], 1)

		if len(_box) > 0:
			box = _box
			text = _text
			result = _result

			for i in range(0, len(box)):
				(startX, startY, endX, endY) = box[i].astype("int")
				(x1, y1, x2, y2) = box[i].astype("int")

				if result[i].startswith( 'unknown' ):
					color = feed['color_unknow']
				else:
					color = feed['color_know']

				r = 10
				d = 10

				y = startY - 10 if startY - 10 > 10 else startY + 10

				cv2.line(overlay, (x1 + r, y1), (x1 + r + d, y1), color, feed['thickness'])
				cv2.line(overlay, (x1, y1 + r), (x1, y1 + r + d), color, feed['thickness'])
				cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, feed['thickness'])

				# Top right drawing
				cv2.line(overlay, (x2 - r, y1), (x2 - r - d, y1), color, feed['thickness'])
				cv2.line(overlay, (x2, y1 + r), (x2, y1 + r + d), color, feed['thickness'])
				cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, feed['thickness'])

				# Bottom left drawing
				cv2.line(overlay, (x1 + r, y2), (x1 + r + d, y2), color, feed['thickness'])
				cv2.line(overlay, (x1, y2 - r), (x1, y2 - r - d), color, feed['thickness'])
				cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, feed['thickness'])

				# Bottom right drawing
				cv2.line(overlay, (x2 - r, y2), (x2 - r - d, y2), color, feed['thickness'])
				cv2.line(overlay, (x2, y2 - r), (x2, y2 - r - d), color, feed['thickness'])
				cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, feed['thickness'])

				cv2.putText(overlay, text[i], (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, feed['thickness'])

				cv2.putText(overlay, result[i], (x1 + r, endY),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, feed['thickness'])

				cv2.line(local['gimage'], (x1 + r, y1), (x1 + r + d, y1), color, feed['thickness'])
				cv2.line(local['gimage'], (x1, y1 + r), (x1, y1 + r + d), color, feed['thickness'])
				cv2.ellipse(local['gimage'], (x1 + r, y1 + r), (r, r), 180, 0, 90, color, feed['thickness'])

				# Top right drawing
				cv2.line(local['gimage'], (x2 - r, y1), (x2 - r - d, y1), color, feed['thickness'])
				cv2.line(local['gimage'], (x2, y1 + r), (x2, y1 + r + d), color, feed['thickness'])
				cv2.ellipse(local['gimage'], (x2 - r, y1 + r), (r, r), 270, 0, 90, color, feed['thickness'])

				# Bottom left drawing
				cv2.line(local['gimage'], (x1 + r, y2), (x1 + r + d, y2), color, feed['thickness'])
				cv2.line(local['gimage'], (x1, y2 - r), (x1, y2 - r - d), color, feed['thickness'])
				cv2.ellipse(local['gimage'], (x1 + r, y2 - r), (r, r), 90, 0, 90, color, feed['thickness'])

				# Bottom right drawing
				cv2.line(local['gimage'], (x2 - r, y2), (x2 - r - d, y2), color, feed['thickness'])
				cv2.line(local['gimage'], (x2, y2 - r), (x2, y2 - r - d), color, feed['thickness'])
				cv2.ellipse(local['gimage'], (x2 - r, y2 - r), (r, r), 0, 0, 90, color, feed['thickness'])

				cv2.putText(local['gimage'], text[i], (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, feed['thickness'])

				cv2.putText(local['gimage'], result[i], (x1 + r, endY),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, feed['thickness'])

	c_fps.update()
	c_fps.stop()
	#print (c_fps.fps())

	#Debug
	if feed['toggleDebug'] is True:
		# grab the frame dimensions
		(h, w) = overlay.shape[:2]
		cv2.putText(overlay, "[DEBUG] elasped time: {:.2f}s".format(c_fps.elapsed()), (10, 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(overlay, "[DEBUG] approx. FPS: {:.2f}".format(c_fps.fps()), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(overlay, "[DEBUG] square detected: {}".format(len(_box)), (10, 45),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(overlay, "[DEBUG] unknown detected: {}".format(_count_unknown), (10, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(overlay, "[DEBUG] total faces: {}".format(len(local['_capture'])), (10, 75),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(overlay, "[DEBUG] recon: {:.2f}s".format(speed), (10, 90),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(overlay, "[DEBUG] background: {}".format(datetime.datetime.now().strftime("%H:%M:%S")), (10, 120),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)

		(h, w) = local['gimage'].shape[:2]
		cv2.putText(local['gimage'], "[DEBUG] elasped time: {:.2f}s".format(c_fps.elapsed()), (10, 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(local['gimage'], "[DEBUG] approx. FPS: {:.2f}".format(c_fps.fps()), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(local['gimage'], "[DEBUG] square detected: {}".format(len(_box)), (10, 45),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(local['gimage'], "[DEBUG] unknown detected: {}".format(_count_unknown), (10, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(local['gimage'], "[DEBUG] total faces: {}".format(len(local['_capture'])), (10, 75),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(local['gimage'], "[DEBUG] recon: {:.2f}s".format(speed), (10, 90),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(local['gimage'], "[DEBUG] background: {}".format(datetime.datetime.now().strftime("%H:%M:%S")), (10, 120),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)

	if feed['banner'] is True:
		cv2.putText(overlay, "{}".format(feed['banner']), (10, h-10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_banner'], 1)
		cv2.putText(local['gimage'], "{}".format(feed['banner']), (10, h-10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_banner'], 1)


	result, local['img_str'] = cv2.imencode('.jpg', overlay)

	return _frames, _last_image, _count_unknown, _box, _text, _result

async def process_video():
	try:
		feed['_camera'] = WebcamVideoStream(src=feed['feedURL']).start()
		await asyncio.sleep(2)
		start_timer = time.time()
		_last_image = np.array([])
		_frames = 0
		_count_unknown = 0
		_added = 0
		_mean = .92
		_meanBackground = 1
		_level = 1
		_text = {}
		_result = ""
		count_unknown = 0
		frames = 0

		_box = np.array([])

		while not local['shutdown']:
			end_timer = time.time()
			if (end_timer - start_timer) > local['timer']:
				if local['_capture']:
					el = min(local['_capture'])
					error = local['_capture'].pop(el, None)
					if error is not None:
						if el.startswith( '0.unknown' ):
							await local['face_recognition'].delete_unknown_names(el)
				start_timer = time.time()

			_frames, _last_image, _count_unknown, _box, _text, _result = await read_frame(_frames, _last_image, _count_unknown, _box, _text, _result)

			await asyncio.sleep(feed['number_fps_second'])

	except asyncio.CancelledError:
		logger.info('stop')
		feed['_camera'].stop()
	except Exception as e:
		logger.info("Error: ", e)

async def mjpeg_video():
	logger.info("mjpeg")
	ws = None
	while ws == None:
		try:
			ws = comm['feed']
		except Exception as e:
			logger.info ("error: {}".format(e))
			await asyncio.sleep(1)
	while True:
		try:
			await ws.send_bytes(local['img_str'].tostring())
		except Exception:
			ws = comm['feed']
		await asyncio.sleep(feed['number_fps_second'])

async def baseHandle(request):
	logger.info ('base handle')
	return web.Response(text="Server "+ fc['hostname'] +" is Up!!!")

async def feedHandle(request):
	logger.info ('feed handle')
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
	try:
			while True:
				result, image = cv2.imencode('.jpg', local['gimage'])
				await write(image.tostring())
				await asyncio.sleep(feed['number_fps_second'])
	except asyncio.CancelledError:
			logger.info("Stream closed by frontend.")
			response = None
	except Exception as e:
		logger.info("Stream close unexptdly, ", e)
	finally:
			if response is not None:
					await response.write_eof()

async def add(_id, photo, firstname, lastname, _oldName):
	newName = _id + '.' + firstname + '_' + lastname
	oldName = '0.' + _oldName
	with open("./known/"+_id+"."+ firstname+"_"+ lastname +".jpg", "wb") as fh:
			fh.write(base64.decodebytes(str.encode(photo)))
	await local['face_recognition'].delete_unknown_names(oldName)
	await local['face_recognition'].unknown_people( None, newName, cv2.imdecode(np.frombuffer(base64.decodebytes(str.encode(photo)), dtype=np.uint8), cv2.IMREAD_COLOR))
	try:
		local['_capture'][newName] = local['_capture'][oldName]
		del local['_capture'][oldName]
		logger.info("Added")
	except Exception as e:
		logger.info ("Unexpected error: {}".format(e))
		pass

#CHANGE KNOWN FOLDER TO VAR
async def delete(_id, firstname, lastname):
	os.remove("./known/" + _id + '.' + firstname + '_' + lastname + ".jpg")
	del local['_capture'][_id + '.' + firstname + '_' + lastname]
	await local['face_recognition'].delete_unknown_names(_id + '.' + firstname + '_' + lastname)
	logger.info ('delete{}'.format(_id))

async def hello(n):
	control = None
	while control == None:
		try:
			control = comm['control']
		except Exception as e:
			logger.info ("error: {}".format(e))
			await asyncio.sleep(5)
	while True:
		try:
			control = comm['control']
		except Exception as e:
			logger.info ("error: {}".format(e))
			asyncio.ensure_future(control())
		logger.info('control heartbeat')
		try:
			await control.send_str(json.dumps({"action":"hello","port":fc['port'],"ip":fc['ip'],"hostname":fc['hostname'],"date":time.strftime("%Y%m%d%H%M%S")}))
		except TimeoutError:
			logger.info("timeout")
		await asyncio.sleep(n)

async def control():
	try:
		await asyncio.sleep(1)
		async with aiohttp.ClientSession() as session:
			async with session.ws_connect(f'ws://control:1880/control') as ws:
		#async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
			#async with session.ws_connect(f'wss://172.10.0.201:1880/control', heartbeat=2, receive_timeout=5) as ws:
				comm['control'] = ws
				logger.info ('control connected')
				async for msg in ws:
					payload = json.loads(msg.data)
					logger.info ("message: {}".format(payload['action']))

					if msg.type == aiohttp.WSMsgType.TEXT:
						if payload['action'] == "debug":
							if payload['id'] == fc['hostname']:
								feed['toggleDebug'] = not feed['toggleDebug']

						elif payload['action'] == "add":
							#logger.debug(payload['id'],payload['firstname'], payload['lastname'],payload['oldName'],payload['treatment'])
							await add(payload['id'],payload['photo'],payload['firstname'], payload['lastname'],payload['oldName'])

						elif payload['action'] == "delete":
							await delete(payload['id'],payload['firstname'], payload['lastname'])

						elif payload['action'] == "start":
							if payload['id'] == fc['hostname']:
								comm['feedler'] = asyncio.ensure_future(process_video())

						elif payload['action'] == "stop":
							if payload['id'] == fc['hostname']:
								comm['feedler'].cancel()
								await comm['feedler']
								await comm['feed'].close()
								comm['feed_task'].cancel()
								await comm['feed_task']

						elif payload['action'] == "quality":
							if payload['id'] == fc['hostname']:
								feed['quality'] = int(payload['value'])

						elif payload['action'] == "fps":
							if payload['id'] == fc['hostname']:
								feed['number_fps_second'] = float(payload['value'])

						elif payload['action'] == "modulo":
							if payload['id'] == fc['hostname']:
								feed['modulo'] = int(payload['value'])

						elif payload['action'] == "mjpeg":
							if payload['id'] == fc['hostname']:
								feed['mjpegToggle'] = int(payload['value'])
								if feed['mjpegToggle'] == 1:
									comm['feed_task'] = asyncio.ensure_future(feedsocket())
									comm['mjpeg_task'] = asyncio.ensure_future(mjpeg_video())
								else:
									comm['mjpeg_task'].cancel()
									await comm['mjpeg_task']
									comm['feedler'].cancel()
									await comm['feedler']

						elif payload['action'] == "size":
							print (payload['id'], fc['hostname'])
							if payload['id'] == fc['hostname']:
								feed['video_size'] = int(payload['value'])

						elif payload['action'] == "feed":
							if payload['id'] == fc['hostname']:
								if  payload['value'] == "0":
									feed['feedURL'] = 0
								else:
									feed['feedURL'] = payload['value']
								fc['feedURL'] = feed['feedURL']
								feed['_camera'].stop()
								feed['_camera'] = WebcamVideoStream(src=feed['feedURL']).start()
						elif msg.type == aiohttp.WSMsgType.CLOSED:
							break
						elif msg.type == aiohttp.WSMsgType.ERROR:
							break
	except asyncio.CancelledError:
		logger.info("Cancelled Task")
	finally:
		logger.info("Rerun control")
		if not local['shutdown']:
			asyncio.ensure_future(control())

async def feedsocket():
	try:
		await asyncio.sleep(1)
		async with aiohttp.ClientSession() as session:
			async with session.ws_connect(f'ws://control:1880/feed2') as ws:
		#async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session2:
			#async with session2.ws_connect(f'wss://api.exception34.com/feed2', heartbeat=2, receive_timeout=5) as ws:
				comm['feed'] = ws
				logger.info ('feed connected')
				async for msg in ws:
					logger.info (msg)
	except asyncio.CancelledError:
		logger.info ("Cancelled Task")
	finally:
		logger.info ("Rerun feed")
		if not local['shutdown']:
			asyncio.ensure_future(feedsocket())

async def on_shutdown(app):
	pass

async def cleanup_background_tasks(app):
	logger.info('cleanup background tasks...')
	try:
		await comm['feed'].close()
	except Exception as e:
		pass
	finally:
		logger.info ("cleanup background tasks completed.")

async def build_server(loop, address, port):
		app = web.Application(loop=loop)

		app.router.add_route('GET', '/feed', feedHandle)
		app.router.add_route('GET', '/', baseHandle)
		#app.on_cleanup.append(cleanup_background_tasks)
		#app.on_shutdown.append(on_shutdown)
		logger.info("Web Server Started!!")
		return await loop.create_server(app.make_handler(), address, port)

if __name__ == '__main__':
		init()
		loop = asyncio.get_event_loop()
		try:
			asyncio.ensure_future(control())
			asyncio.ensure_future(hello(60))
			comm['feedler'] = asyncio.ensure_future(process_video())

			loop.run_until_complete(build_server(loop, '0.0.0.0', 5000))
			logger.info("Camera Script Ready!")

			loop.run_forever()
		except KeyboardInterrupt:
			 local['shutdown'] = True
		finally:
			afile = open(r'./.config', 'wb')
			print (fc)
			pickle.dump(fc, afile)
			afile.close()

			logger.info("Shutting down {} tasks".format(len(asyncio.Task.all_tasks())))
			for task in asyncio.Task.all_tasks():
					print(f'Cancelling: {task}')
			for task in asyncio.Task.all_tasks():
					task.cancel()
					loop.run_until_complete(task)

			loop.close()
			logger.info("Shutting Down!")
			sys.exit(1)


