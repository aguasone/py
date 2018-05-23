import asyncio
import os
import aiohttp
import json
import subprocess

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


#Face Recognition parameters
known_people_folder = 'knownall'
tolerance = 0.68
show_distance = True
prototxt = "deploy.prototxt.txt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
shape_predictor = "shape_predictor_68_face_landmarks.dat"
confidence = 0.8


# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# print("[INFO] loading shape-predictor...")
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])
# fa = FaceAligner(predictor, desiredFaceWidth=256)

start_encoding = time.time()
face_recognition = Recon(known_people_folder, tolerance, show_distance)
print('encoding of the known folder completed in: {:.2f}s'.format(time.time() - start_encoding))

toggleDebug = True
modulo = 2
color_unknow = (0, 0, 255)
color_known = (162, 255, 0)
thickness = 2
number_fps_second = 1/25
quality = 10
banner = "hello..."
color_banner = (0, 0, 255)

ffmpeg_feed = False

host = os.getenv('HOST', '0.0.0.0')
port = int(os.getenv('PORT', 5000))

async def testhandle(request):
	return web.Response(text='Test handle')

async def read_frame(image, _capture, _frames, _last_image, _count_unknown, _box, _text, _result):
	err = 0
	check = 0
	squares = 0
	speed = 0
	score = 0
	_frames = _frames + 1

	c_fps = FPS().start()
	overlay = imutils.resize(image, width=600)

	if len(_last_image) > 0:
		err = np.sum((overlay.astype("float") - _last_image.astype("float")) ** 2)
		err /= float(overlay.shape[0] * overlay.shape[1])
		#print (err)

	if (_frames % modulo == 0):

	# Make copies of the frame for transparency processing
	#overlay = frame.copy()
	#output = overlay.copy()

	#set transparency value
	#alpha  = 0.5

	# make semi-transparent bounding box
	#cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

	# if score < _meanBackground:

	# grab the frame dimensions and convert it to a blob
		(h, w) = overlay.shape[:2]

		blob = cv2.dnn.blobFromImage(cv2.resize(overlay, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()


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
			if _confidence < confidence:
				continue

			squares = squares + 1
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box[i] = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box[i].astype("int")
			(x1, y1, x2, y2) = box[i].astype("int")

			# draw the bounding box of the face along with the associated
			# probability
			txt = "{:.2f}%".format(confidence * 100)
			text[i] = txt
			y = startY - 10 if startY - 10 > 10 else startY + 10

			#gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
			cropped = overlay[startY:endY, startX:endX]

			start_encoding = time.time()
			found = False
			array = []
			distances = []
			array, distances= await face_recognition.test_image(cropped)

			result[i] = 'unknown'
			if array != ['unknown']:
				# If Customer Known
				try:
					_capture[array[0]]
				except KeyError:
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
					#socketIO.emit('add', json.dumps({"name": array[0], "confidence":text[i], "image": base64_bytes.decode('utf-8') }))#, "image":  cropped.tolist()}))
				if array[0].startswith( 'unknown' ):
					color = color_unknow
				else:
					color = color_known
				result[i] = array[0]
			else:
				#If Customer unknown
				color = color_unknow
				add = face_recognition.unknown_people(cropped, 'unknown'+str(_count_unknown), time.strftime("%Y-%m-%d_%H%M%S"), box[i])
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
					_capture['unknown'+str(_count_unknown)] = [cropped, box[i], text[i]]

					retval, jpg = cv2.imencode('.jpg', saved)
					base64_bytes = b64encode(jpg)
					#socketIO.emit('add', json.dumps({"name": 'unknown'+str(_count_unknown)+"_"+ time.strftime("%Y-%m-%d_%H%M%S") +".png", "confidence":text[i], "image": base64_bytes.decode('utf-8') }))#, "image":  cropped.tolist()}))
					array[0] = 'unknown'+str(_count_unknown)
					result[i] = str(array[0])
				_count_unknown = _count_unknown + 1

			speed = time.time() - start_encoding


			r = 10
			d = 10

			cv2.line(overlay, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
			cv2.line(overlay, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
			cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

			# Top right drawing
			cv2.line(overlay, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
			cv2.line(overlay, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
			cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

			# Bottom left drawing
			cv2.line(overlay, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
			cv2.line(overlay, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
			cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

			# Bottom right drawing
			cv2.line(overlay, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
			cv2.line(overlay, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
			cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

			cv2.putText(overlay, text[i], (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness)

			cv2.putText(overlay, result[i], (x1 + r, endY),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness)

		_box = box
		_text = text
		_result = result

		# else:
		#   if len(_box) > 0:
		#     box = _box
		#     text = _text
		#     result = _result

		#     for i in range(0, len(box)):
		#       (startX, startY, endX, endY) = box[i].astype("int")
		#       (x1, y1, x2, y2) = box[i].astype("int")

		#       if result[i].startswith( 'unknown' ):
		#         color = (0, 0, 255)
		#       else:
		#         color = (162, 255, 0)

		#       thickness = 2
		#       r = 10
		#       d = 10

		#       y = startY - 10 if startY - 10 > 10 else startY + 10

		#       cv2.line(overlay, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
		#       cv2.line(overlay, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
		#       cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

		#       # Top right drawing
		#       cv2.line(overlay, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
		#       cv2.line(overlay, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
		#       cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

		#       # Bottom left drawing
		#       cv2.line(overlay, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
		#       cv2.line(overlay, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
		#       cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

		#       # Bottom right drawing
		#       cv2.line(overlay, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
		#       cv2.line(overlay, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
		#       cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

		#       cv2.putText(overlay, text[i], (startX, y),
		#         cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness)

		#       cv2.putText(overlay, result[i], (x1 + r, endY),
		#         cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness)

	else:
		if len(_box) > 0:
			box = _box
			text = _text
			result = _result

			for i in range(0, len(box)):
				(startX, startY, endX, endY) = box[i].astype("int")
				(x1, y1, x2, y2) = box[i].astype("int")

				if result[i].startswith( 'unknown' ):
					color = color_unknow
				else:
					color = color_known

				r = 10
				d = 10

				y = startY - 10 if startY - 10 > 10 else startY + 10

				cv2.line(overlay, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
				cv2.line(overlay, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
				cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

				# Top right drawing
				cv2.line(overlay, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
				cv2.line(overlay, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
				cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

				# Bottom left drawing
				cv2.line(overlay, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
				cv2.line(overlay, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
				cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

				# Bottom right drawing
				cv2.line(overlay, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
				cv2.line(overlay, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
				cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

				cv2.putText(overlay, text[i], (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness)

				cv2.putText(overlay, result[i], (x1 + r, endY),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness)

	c_fps.update()
	c_fps.stop()
	#print (c_fps.fps())

	#Debug
	if toggleDebug is True:
		# grab the frame dimensions
		(h, w) = overlay.shape[:2]
		cv2.putText(overlay, "[DEBUG] elasped time: {:.2f}s".format(c_fps.elapsed()), (10, 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
		cv2.putText(overlay, "[DEBUG] approx. FPS: {:.2f}".format(c_fps.fps()), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
		cv2.putText(overlay, "[DEBUG] square detected: {}".format(len(_box)), (10, 45),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
		cv2.putText(overlay, "[DEBUG] unknown detected: {}".format(_count_unknown), (10, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
		cv2.putText(overlay, "[DEBUG] active faces: {}".format(len(_capture)), (10, 75),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
		cv2.putText(overlay, "[DEBUG] recon: {:.2f}s".format(speed), (10, 90),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
		cv2.putText(overlay, "[DEBUG] added faces: {}".format(len(_capture)), (10, 105),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
		cv2.putText(overlay, "[DEBUG] background: {:}".format(err), (10, 120),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
		cv2.putText(overlay, "{}".format(banner), (10, h-10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_banner, 1)


	_last_image = overlay

	return overlay, _capture, _frames, _last_image, _count_unknown, _box, _text, _result

async def feedHandle(request):
	print ('feed handle')
	# if request.headers.get('accept') != 'text/event-stream':
	#     return web.Response(status=406)

	ws = request.app['feed']

	if ffmpeg_feed:
		stream = request.content
	else:
		_camera = WebcamVideoStream(src=0).start()

	print (request.content)
	time.sleep(2.0)
	_last_image = np.array([])
	_frames = 0
	_capture = {}
	_count_unknown = 0
	_added = 0
	_mean = .92
	_meanBackground = 1
	_level = 1
	_text = {}
	_result = ""

	try:
		data = b''
		capture = {}
		count_unknown = 0
		frames = 0

		_box = np.array([])

		if ffmpeg_feed:
			while True:
				time.sleep(number_fps_second)
				chunk = await stream.read(102400)
				data += chunk
				if not chunk:
					break
				jpg_start = data.find(b'\xff\xd8')
				jpg_end = data.find(b'\xff\xd9')
				if jpg_start != -1 and jpg_end != -1:
					image = data[jpg_start:jpg_end + 2]
					image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)

					image, _capture, _frames, _last_image, _count_unknown, _box, _text, _result = await read_frame(image, _capture, _frames, _last_image, _count_unknown, _box, _text, _result)

					encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
					result, img_str = cv2.imencode('.jpg', image, encode_param)

					try:
						await ws.send_bytes(img_str.tostring())
					except Exception:
						ws = request.app['feed']
				# data = data[jpg_end + 2:]
		else:
			while True:
				time.sleep(number_fps_second)
				image = _camera.read()
				if not image.any():
					break
				image, _capture, _frames, _last_image, _count_unknown, _box, _text, _result = await read_frame(image, _capture, _frames, _last_image, _count_unknown, _box, _text, _result)
				encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
				result, img_str = cv2.imencode('.jpg', image, encode_param)
				try:
					await ws.send_bytes(img_str.tostring())
				except Exception:
					ws = request.app['feed']
	except asyncio.CancelledError:
		await request.close()
		print ('end handle')
	#finally:
	#if web.Response is not None:
		#await web.Response.write_eof()

async def callback(msg):
	#app = App()
	#await app.decode_message('{"action": "stream", "payload": true}')
	print(msg)

async def websocket(app):
	try:
		session = aiohttp.ClientSession()
		async with session.ws_connect(f'ws://exception34.com:1880/control') as ws:
			app['ws'] = ws
			print ('ws connected')
			async for msg in ws:
				payload = json.loads(msg.data)
				await ws.send_str('hello')
				await callback(msg.data)
				if msg.type == aiohttp.WSMsgType.TEXT:
					if payload['payload'] == "debug":
						await callback(payload['payload'])
					elif payload['payload'] == "add":
						await callback(payload['payload'])
					elif msg.type == aiohttp.WSMsgType.CLOSED:
						break
					elif msg.type == aiohttp.WSMsgType.ERROR:
						break
	except Exception:
		pass
	finally:
		app['websocket_task'] = app.loop.create_task(websocket(app))

async def feedsocket(app):
	try:
		session2 = aiohttp.ClientSession()
		async with session2.ws_connect(f'ws://exception34.com:1880/feed2') as ws:
			app['feed'] = ws
			print ('feed connected')
			async for msg in ws:
				print (msg)
	except Exception:
		pass
	finally:
		app['feed_task'] = app.loop.create_task(feedsocket(app))

async def init_ws(app):
	app['websocket_task'] = app.loop.create_task(websocket(app))

async def init_feed(app):
	app['feed_task'] = app.loop.create_task(feedsocket(app))

async def on_shutdown(app):
	print ('shutdown')

async def cleanup_background_tasks(app):
	print('cleanup background tasks...')
	await app['ws'].close()
	await app['feed'].close()
	app['websocket_task'].cancel()
	app['feed_task'].cancel()
	await app['websocket_task']
	await app['feed_task']


def main():
	app = web.Application()
	app.router.add_route('GET', '/', testhandle)
	app.router.add_route('POST', '/feed', feedHandle)
	app.on_startup.append(init_feed)
	app.on_startup.append(init_ws)
	app.on_cleanup.append(cleanup_background_tasks)
	app.on_shutdown.append(on_shutdown)
	web.run_app(app, host=host, port=port)

if __name__ == '__main__':
	main()
