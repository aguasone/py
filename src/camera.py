import asyncio
import os
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

#LOGS
formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
logging.basicConfig(format="[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s")

# create a file handler
handler = logging.FileHandler('camera.log')
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger('async')
logger.setLevel(logging.INFO)


fc = {}
feed = {}
_capture = {}
timer = 60
shutdown = False

host = os.getenv('HOST', '0.0.0.0')
port = int(os.getenv('PORT', 5000))


async def init(app):

	#Face Recognition parameters
	fc['known_people_folder'] = 'known'
	fc['tolerance'] = 0.68
	fc['show_distance'] = True
	fc['prototxt'] = "deploy.prototxt.txt"
	fc['model'] = "res10_300x300_ssd_iter_140000.caffemodel"
	fc['shape_predictor'] = "shape_predictor_68_face_landmarks.dat"
	fc['confidence'] = 0.8


	# print("[INFO] loading shape-predictor...")
	# detector = dlib.get_frontal_face_detector()
	# predictor = dlib.shape_predictor(args["shape_predictor"])
	# fa = FaceAligner(predictor, desiredFaceWidth=256)

	feed['toggleDebug'] = True
	feed['banner'] = False

	feed['modulo'] = 2
	feed['color_unknow'] = (0, 0, 255)
	feed['color_know'] = (162, 255, 0)
	feed['thickness'] = 2
	feed['number_fps_second'] = 1/25
	feed['quality'] = 5
	feed['banner'] = "hello..."
	feed['color_banner'] = (0, 0, 255)

	feed['video_size'] = 600

	#ffmpeg -f avfoundation -framerate 30 -i "0" -f mjpeg -q:v 100 -s 640x480 http://localhost:5000/feed
	#ffmpeg -re -i ../../sabnzbd/complete/shows/The.Big.Bang.Theory.S11E22.mkv -f mjpeg -q:v 20 -s 640x480 http://localhost:5000/feed
	#TODO: add ffmpeg internally

	feed['ffmpeg_feed'] = False

	# load our serialized model from disk
	logger.info("[INFO] loading model...")
	fc['net'] = cv2.dnn.readNetFromCaffe(fc['prototxt'], fc['model'])

	start_encoding = time.time()
	fc['face_recognition'] = Recon(fc['known_people_folder'], fc['tolerance'], fc['show_distance'])
	logger.info('encoding of the known folder completed in: {:.2f}s'.format(time.time() - start_encoding))

	#Check if OpenCV is Optimized:
	logger.info ("CV2 Optimized: {}".format(cv2.useOptimized()))

	# #reload object from file
	# file2 = open(r'C:\d.pkl', 'rb')
	# new_d = pickle.load(file2)
	# file2.close()

async def stopHandle(request):
	shutdown = True
	return web.Response(text='Shutting down...')

async def read_frame(_camera, _frames, _last_image, _count_unknown, _box, _text, _result, app, _size):
	err = 0
	check = 0
	squares = 0
	speed = 0
	score = 0
	_frames = _frames + 1
	global gimage
	global img_str

	c_fps = FPS().start()
	image = _camera.read()
	overlay = imutils.resize(image, width=_size)
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
		fc['net'].setInput(blob)
		detections = fc['net'].forward()


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
			array, distances = await fc['face_recognition'].test_image(cropped)
			logger.debug("array: {}".format(array))
			result[i] = 'unknown'
			if (array != ['0.unknown']) and (array != []):
				# If Customer Known
				try:
					_capture[array[0]]
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

					_capture[array[0]] = [cropped, box[i], text[i]]

					retval, jpg = cv2.imencode('.jpg', saved)
					base64_bytes = b64encode(jpg)
					control = None
					while control == None:
						try:
							control = app['control']
							logger.info('add known: {}'.format(str.split(array[0], '.')[0]))
							await control.send_str(json.dumps({"action":"add_known", "date": time.strftime("%Y%m%d%H%M%S"), "name": str.split(array[0], '.')[0], "confidence":text[i], "image": base64_bytes.decode('utf-8') }))#, "image":  cropped.tolist()}))
						except Exception as e:
							logger.info("ws error: ", e)
				if array[0].startswith( '0.unknown' ):
					color = feed['color_unknow']
				else:
					color = feed['color_know']
				result[i] = str.split(array[0], '.')[1]
			else:
				#If Customer unknown
				color = feed['color_unknow']
				add = await fc['face_recognition'].unknown_people(cropped, '0.unknown'+str(_count_unknown), None)
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
					_capture['0.unknown'+str(_count_unknown)] = [cropped, box[i], text[i]]
					retval, jpg = cv2.imencode('.jpg', saved)
					base64_bytes = b64encode(jpg)
					control = None
					while control == None:
						try:
							control = app['control']
							logger.info('add unknown: {}'.format('unknown'+str(_count_unknown)))
							await control.send_str(json.dumps({"action":"add_unknown", "date": time.strftime("%Y%m%d%H%M%S"), "name": 'unknown'+str(_count_unknown), "confidence":text[i], "image": base64_bytes.decode('utf-8') }))#, "image":  cropped.tolist()}))
						except Exception as e:
							logger.info("error: ", e)

					result[i] = 'unknown'+str(_count_unknown)
				_count_unknown = _count_unknown + 1

			speed = time.time() - start_encoding

			r = 10
			d = 10

			score = 1
			_last_image = overlay
			gimage = overlay.copy()
			encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), feed['quality']]
			rslt, img_str = cv2.imencode('.jpg', overlay, encode_param)
			overlay = cv2.imdecode(img_str, 1)

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

			cv2.line(gimage, (x1 + r, y1), (x1 + r + d, y1), color, feed['thickness'])
			cv2.line(gimage, (x1, y1 + r), (x1, y1 + r + d), color, feed['thickness'])
			cv2.ellipse(gimage, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, feed['thickness'])

			# Top right drawing
			cv2.line(gimage, (x2 - r, y1), (x2 - r - d, y1), color, feed['thickness'])
			cv2.line(gimage, (x2, y1 + r), (x2, y1 + r + d), color, feed['thickness'])
			cv2.ellipse(gimage, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, feed['thickness'])

			# Bottom left drawing
			cv2.line(gimage, (x1 + r, y2), (x1 + r + d, y2), color, feed['thickness'])
			cv2.line(gimage, (x1, y2 - r), (x1, y2 - r - d), color, feed['thickness'])
			cv2.ellipse(gimage, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, feed['thickness'])

			# Bottom right drawing
			cv2.line(gimage, (x2 - r, y2), (x2 - r - d, y2), color, feed['thickness'])
			cv2.line(gimage, (x2, y2 - r), (x2, y2 - r - d), color, feed['thickness'])
			cv2.ellipse(gimage, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, feed['thickness'])

			cv2.putText(gimage, text[i], (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, feed['thickness'])

			cv2.putText(gimage, result[i], (x1 + r, endY),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, feed['thickness'])

		_box = box
		_text = text
		_result = result
		_last_image = overlay
		if score != 1:
			gimage = overlay.copy()
			encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), feed['quality']]
			rslt, img_str = cv2.imencode('.jpg', overlay, encode_param)
			overlay = cv2.imdecode(img_str, 1)

	else:
		_last_image = overlay
		gimage = overlay.copy()
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), feed['quality']]
		rslt, img_str = cv2.imencode('.jpg', overlay, encode_param)
		overlay = cv2.imdecode(img_str, 1)

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

				cv2.line(gimage, (x1 + r, y1), (x1 + r + d, y1), color, feed['thickness'])
				cv2.line(gimage, (x1, y1 + r), (x1, y1 + r + d), color, feed['thickness'])
				cv2.ellipse(gimage, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, feed['thickness'])

				# Top right drawing
				cv2.line(gimage, (x2 - r, y1), (x2 - r - d, y1), color, feed['thickness'])
				cv2.line(gimage, (x2, y1 + r), (x2, y1 + r + d), color, feed['thickness'])
				cv2.ellipse(gimage, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, feed['thickness'])

				# Bottom left drawing
				cv2.line(gimage, (x1 + r, y2), (x1 + r + d, y2), color, feed['thickness'])
				cv2.line(gimage, (x1, y2 - r), (x1, y2 - r - d), color, feed['thickness'])
				cv2.ellipse(gimage, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, feed['thickness'])

				# Bottom right drawing
				cv2.line(gimage, (x2 - r, y2), (x2 - r - d, y2), color, feed['thickness'])
				cv2.line(gimage, (x2, y2 - r), (x2, y2 - r - d), color, feed['thickness'])
				cv2.ellipse(gimage, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, feed['thickness'])

				cv2.putText(gimage, text[i], (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, feed['thickness'])

				cv2.putText(gimage, result[i], (x1 + r, endY),
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
		cv2.putText(overlay, "[DEBUG] total faces: {}".format(len(_capture)), (10, 75),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(overlay, "[DEBUG] recon: {:.2f}s".format(speed), (10, 90),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(overlay, "[DEBUG] background: {}".format(datetime.datetime.now().strftime("%H:%M:%S")), (10, 120),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)

		(h, w) = gimage.shape[:2]
		cv2.putText(gimage, "[DEBUG] elasped time: {:.2f}s".format(c_fps.elapsed()), (10, 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(gimage, "[DEBUG] approx. FPS: {:.2f}".format(c_fps.fps()), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(gimage, "[DEBUG] square detected: {}".format(len(_box)), (10, 45),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(gimage, "[DEBUG] unknown detected: {}".format(_count_unknown), (10, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(gimage, "[DEBUG] total faces: {}".format(len(_capture)), (10, 75),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(gimage, "[DEBUG] recon: {:.2f}s".format(speed), (10, 90),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)
		cv2.putText(gimage, "[DEBUG] background: {}".format(datetime.datetime.now().strftime("%H:%M:%S")), (10, 120),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_know'], 1)

	if feed['banner'] is True:
		cv2.putText(overlay, "{}".format(feed['banner']), (10, h-10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_banner'], 1)
		cv2.putText(gimage, "{}".format(feed['banner']), (10, h-10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, feed['color_banner'], 1)


	result, img_str = cv2.imencode('.jpg', overlay)

	return img_str, _frames, _last_image, _count_unknown, _box, _text, _result

async def process_video(app):
	try:
		_camera = WebcamVideoStream(src=0).start()
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
		this_size = 600 #feed['video_size']
		count_unknown = 0
		frames = 0
		control = None
		while control == None:
			try:
				control = app['control']
			except Exception as e:
				logger.info ("error: {}".format(e))
				await asyncio.sleep(1)

		_box = np.array([])

		while True:
			end_timer = time.time()
			if (end_timer - start_timer) > timer:
				if _capture:
					el = min(_capture)
					error = _capture.pop(el, None)
					if error is not None:
						if el.startswith( '0.unknown' ):
							await fc['face_recognition'].delete_unknown_names(el)
				start_timer = time.time()
			if shutdown:
				break

			image, _frames, _last_image, _count_unknown, _box, _text, _result = await read_frame(_camera, _frames, _last_image, _count_unknown, _box, _text, _result, app, this_size)

			await asyncio.sleep(feed['number_fps_second'])

	except asyncio.CancelledError:
		logger.info('stop')
		_camera.stop()
	except Exception as e:
		logger.info("Error: ", e)

async def mjpeg_video(app):
	logger.info("mjpeg")
	ws = None
	while ws == None:
		try:
			ws = app['feed']
		except Exception as e:
			logger.info ("error: {}".format(e))
			await asyncio.sleep(1)
	global img_str
	while True:
		try:
			await ws.send_bytes(img_str.tostring())
		except Exception:
			ws = app['feed']
		await asyncio.sleep(feed['number_fps_second'])

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
	global gimage

	try:
			while True:
				result, image = cv2.imencode('.jpg', gimage)
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
	with open("./knownall/"+_id+"."+ firstname+"_"+ lastname +".jpg", "wb") as fh:
			fh.write(base64.decodebytes(str.encode(photo)))
	await fc['face_recognition'].delete_unknown_names(oldName)
	await fc['face_recognition'].unknown_people( None, newName, cv2.imdecode(np.frombuffer(base64.decodebytes(str.encode(photo)), dtype=np.uint8), cv2.IMREAD_COLOR))
	try:
		_capture[newName] = _capture[oldName]
		del _capture[oldName]
		logger.info("Added")
	except Exception as e:
		logger.info ("Unexpected error: {}".format(e))
		pass

async def delete(_id, firstname, lastname):
	os.remove("./knownall/" + _id + '.' + firstname + '_' + lastname + ".jpg")
	del _capture[_id + '.' + firstname + '_' + lastname]
	await fc['face_recognition'].delete_unknown_names(_id + '.' + firstname + '_' + lastname)
	logger.info ('delete{}'.format(_id))

async def control(app):
	try:
		#session = aiohttp.ClientSession()
		#async with session.ws_connect(f'ws://localhost:1880/control') as ws:
		async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
			async with session.ws_connect(f'wss://api.exception34.com/control', heartbeat=1, receive_timeout=5) as ws:
				app['control'] = ws
				logger.info ('ws connected')
				async for msg in ws:
					payload = json.loads(msg.data)
					logger.info ("message: {}".format(payload['action']))
					#await ws.send_str('hello')
					#await callback(msg.data)
					if msg.type == aiohttp.WSMsgType.TEXT:
						if payload['action'] == "debug":
							await callback(payload['payload'])
						elif payload['action'] == "add":
							await add(payload['id'],payload['photo'],payload['firstname'], payload['lastname'],payload['oldName'])
						elif payload['action'] == "delete":
							await delete(payload['id'],payload['firstname'], payload['lastname'])
						elif payload['action'] == "start":
							await start_feed(app)
						elif payload['action'] == "stop":
							app['feedler'].cancel()
							await app['feedler']
							await app['feed'].close()
							app['feed_task'].cancel()
							await app['feed_task']
						elif payload['action'] == "feed['quality']":
							feed['quality'] = int(payload['value'])
						elif payload['action'] == "fps":
							feed['number_fps_second'] = float(payload['value'])
						elif payload['action'] == "modulo":
							feed['modulo'] = int(payload['value'])
						elif payload['action'] == "size":
							feed['video_size'] = int(payload['value'])
							if feed['video_size'] == 100:
								await init_feed(app)
								await init_mjpeg(app)
							if feed['video_size'] == 600:
								app['mjpeg_task'].cancel()
								await app['mjpeg_task']
								app['feedler'].cancel()
								await app['feedler']

						elif msg.type == aiohttp.WSMsgType.CLOSED:
							break
						elif msg.type == aiohttp.WSMsgType.ERROR:
							break
	except Exception as e:
		logger.info ("Unexpected error: {}".format(e))
		pass
	finally:
		app['control_task'] = app.loop.create_task(control(app))

async def feedsocket(app):
	try:
		#session2 = aiohttp.ClientSession()
		#async with session2.ws_connect(f'ws://localhost:1880/feed2') as ws:
		session2 = aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False))
		async with session2.ws_connect(f'wss://api.exception34.com/feed2', heartbeat=1, receive_timeout=5) as ws:
			app['feed'] = ws
			logger.info ('feed connected')
			async for msg in ws:
				logger.info (msg)
	except Exception:
		pass
	finally:
		app['feed_task'] = app.loop.create_task(feedsocket(app))

async def init_mjpeg(app):
	app['mjpeg_task'] = app.loop.create_task(mjpeg_video(app))

async def init_control(app):
	app['control_task'] = app.loop.create_task(control(app))

async def init_feed(app):
	app['feed_task'] = app.loop.create_task(feedsocket(app))

async def start_feed(app):
	app['feedler'] = app.loop.create_task(process_video(app))

async def on_shutdown(app):
	logger.info ('shutdown')
	afile = open(r'./.config', 'wb')
	pickle.dump(prototxt, afile)
	afile.close()

async def cleanup_background_tasks(app):
	logger.info('cleanup background tasks...')
	await app['control'].close()
	app['control_task'].cancel()
	try:
		await app['feed'].close()
		app['feed_task'].cancel()
	except Exception as e:
		pass
	finally:
		logger.info ("cleanup background tasks completed.")

def main():
	app = web.Application()
	app.on_startup.append(init)

	app.router.add_route('GET', '/feed', feedHandle)

	app.on_startup.append(init_control)

	app.on_cleanup.append(cleanup_background_tasks)
	app.on_shutdown.append(on_shutdown)

	web.run_app(app, host=host, port=port)

if __name__ == '__main__':
	main()
