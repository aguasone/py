import asyncio
import os
import aiohttp
import json
import subprocess
import inspect

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
_capture = {}


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
banner = False
shutdown = False

modulo = 5
color_unknow = (0, 0, 255)
color_known = (162, 255, 0)
thickness = 2
number_fps_second = 0 #1/25
quality = 25
banner = "hello..."
color_banner = (0, 0, 255)
timer = 20

#ffmpeg -f avfoundation -framerate 30 -i "0" -f mjpeg -q:v 100 -s 640x480 http://localhost:5000/feed
#ffmpeg -re -i ../../sabnzbd/complete/shows/The.Big.Bang.Theory.S11E22.mkv -f mjpeg -q:v 20 -s 640x480 http://localhost:5000/feed
#TODO: add ffmpeg internally

ffmpeg_feed = True

host = os.getenv('HOST', '0.0.0.0')
port = int(os.getenv('PORT', 5000))

#Check if OpenCV is Optimized:
print ("CV2 Optimized: ", cv2.useOptimized())

async def testhandle(request):
	return web.Response(text='Test handle')

async def stopHandle(request):
	shutdown = True
	return web.Response(text='Shutting down...')

async def read_frame(image, _frames, _last_image, _count_unknown, _box, _text, _result, control):
	err = 0
	check = 0
	squares = 0
	speed = 0
	score = 0
	_frames = _frames + 1

	c_fps = FPS().start()
	overlay = imutils.resize(image, width=600)
	overlay = cv2.flip( overlay, 1 )

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
			txt = "{:.2f}%".format(_confidence * 100)
			text[i] = txt
			y = startY - 10 if startY - 10 > 10 else startY + 10

			#gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
			cropped = overlay[startY:endY, startX:endX]

			start_encoding = time.time()
			found = False
			array = []
			distances = []
			array, distances = await face_recognition.test_image(cropped)
			#print ("array: ", array)
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
					await control.send_str(json.dumps({"action":"add_known", "date": time.strftime("%Y%m%d%H%M%S"), "name": str.split(array[0], '.')[0], "confidence":text[i], "image": base64_bytes.decode('utf-8') }))#, "image":  cropped.tolist()}))
				if array[0].startswith( '0.unknown' ):
					color = color_unknow
				else:
					color = color_known
				result[i] = str.split(array[0], '.')[1]
			else:
				#If Customer unknown
				color = color_unknow
				add = await face_recognition.unknown_people(cropped, '0.unknown'+str(_count_unknown), None)
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
					await control.send_str(json.dumps({"action":"add_unknown", "date": time.strftime("%Y%m%d%H%M%S"), "name": 'unknown'+str(_count_unknown), "confidence":text[i], "image": base64_bytes.decode('utf-8') }))#, "image":  cropped.tolist()}))
					result[i] = 'unknown'+str(_count_unknown)
				_count_unknown = _count_unknown + 1

			speed = time.time() - start_encoding

			r = 10
			d = 10

			_last_image = overlay
			encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
			rslt, img_str = cv2.imencode('.jpg', overlay, encode_param)
			overlay = cv2.imdecode(img_str, 1)

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

	else:
		_last_image = overlay
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
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
		cv2.putText(overlay, "[DEBUG] total faces: {}".format(len(_capture)), (10, 75),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
		cv2.putText(overlay, "[DEBUG] recon: {:.2f}s".format(speed), (10, 90),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
		cv2.putText(overlay, "[DEBUG] background: {:}".format(err), (10, 120),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

	if banner is True:
		cv2.putText(overlay, "{}".format(banner), (10, h-10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_banner, 1)

	return overlay, _frames, _last_image, _count_unknown, _box, _text, _result

async def feedHandle(request):
	print ('feed handle')
	# if request.headers.get('accept') != 'text/event-stream':
	#     return web.Response(status=406)

	ws = request.app['feed']
	control = request.app['control']

	if ffmpeg_feed:
		stream = request.content
	else:
		_camera = WebcamVideoStream(src=0).start()

	print (request.content)
	time.sleep(2.0)
	_last_image = np.array([])
	_frames = 0
	_count_unknown = 0
	_added = 0
	_mean = .92
	_meanBackground = 1
	_level = 1
	_text = {}
	_result = ""
	start_timer = time.time()


	try:
		data = b''
		count_unknown = 0
		frames = 0

		_box = np.array([])

		if ffmpeg_feed:
			while True:
				end_timer = time.time()
				if (end_timer - start_timer) > timer:
					global _capture
					if _capture:
						el = min(_capture)
						error = _capture.pop(el, None)
						print (inspect.currentframe().f_lineno, " ", el)
						if error is not None:
							if el.startswith( '0.unknown' ):
								await face_recognition.delete_unknown_names(el)
					start_timer = time.time()
				if shutdown:
					break
				time.sleep(number_fps_second)
				chunk = await stream.read(102400)
				if not chunk:
					break
				data += chunk
				jpg_start = data.find(b'\xff\xd8')
				jpg_end = data.find(b'\xff\xd9')
				if jpg_start != -1 and jpg_end != -1:
					image = data[jpg_start:jpg_end + 2]
					image = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)

					image, _frames, _last_image, _count_unknown, _box, _text, _result = await read_frame(image, _frames, _last_image, _count_unknown, _box, _text, _result, control)

					result, img_str = cv2.imencode('.jpg', image)

					try:
						await ws.send_bytes(img_str.tostring())
					except Exception:
						ws = request.app['feed']
					data = data[jpg_end + 2:]
		else:
			while True:
				if (time.time() - start_timer) > timer:
					_capture = {}
					start_timer = time.time()
				if shutdown:
					break
				time.sleep(number_fps_second)
				image = _camera.read()
				if not image.any():
					break
				image, _frames, _last_image, _count_unknown, _box, _text, _result = await read_frame(image, _frames, _last_image, _count_unknown, _box, _text, _result, control)

				result, img_str = cv2.imencode('.jpg', image)

				try:
					await ws.send_bytes(img_str.tostring())
				except Exception:
					ws = request.app['feed']
	except Exception as e:
		await request.close()
		print ('end handle: ', e)
	finally:
		if web.Response is not None:
			print ("todo")
			#await web.Response.write_eof()

async def add(_id, photo, firstname, lastname, _oldName):
	newName = _id + '.' + firstname + '_' + lastname
	oldName = '0.' + _oldName
	with open("./knownall/"+_id+"."+ firstname+"_"+ lastname +".jpg", "wb") as fh:
			fh.write(base64.decodebytes(str.encode(photo)))
	await face_recognition.delete_unknown_names(oldName)
	await face_recognition.unknown_people( None, newName, cv2.imdecode(np.frombuffer(base64.decodebytes(str.encode(photo)), dtype=np.uint8), cv2.IMREAD_COLOR))
	try:
		_capture[newName] = _capture[oldName]
		del _capture[oldName]
		print("Added")
	except Exception as e:
		print ("Unexpected error:", e)
		pass

async def delete(_id, firstname, lastname):
	os.remove("./knownall/" + _id + '.' + firstname + '_' + lastname + ".jpg")
	del _capture[_id + '.' + firstname + '_' + lastname]
	await face_recognition.delete_unknown_names(_id + '.' + firstname + '_' + lastname)
	print ('delete', _id)

async def websocket(app):
	try:
		session = aiohttp.ClientSession()
		async with session.ws_connect(f'ws://localhost:1880/control') as ws:
			app['control'] = ws
			print ('ws connected')
			async for msg in ws:
				payload = json.loads(msg.data)
				print ("message:", payload['action'])
				#await ws.send_str('hello')
				#await callback(msg.data)
				if msg.type == aiohttp.WSMsgType.TEXT:
					if payload['action'] == "debug":
						await callback(payload['payload'])
					elif payload['action'] == "add":
						await add(payload['id'],payload['photo'],payload['firstname'], payload['lastname'],payload['oldName'])
					elif payload['action'] == "delete":
						await delete(payload['id'],payload['firstname'], payload['lastname'])
					elif msg.type == aiohttp.WSMsgType.CLOSED:
						break
					elif msg.type == aiohttp.WSMsgType.ERROR:
						break
	except Exception as e:
		print ("Unexpected error:", e)
		pass
	finally:
		app['websocket_task'] = app.loop.create_task(websocket(app))

async def feedsocket(app):
	try:
		session2 = aiohttp.ClientSession()
		async with session2.ws_connect(f'ws://localhost:1880/feed2') as ws:
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
	await app['control'].close()
	await app['feed'].close()
	app['websocket_task'].cancel()
	app['feed_task'].cancel()
	await app['websocket_task']
	await app['feed_task']


def main():
	app = web.Application()
	app.router.add_route('GET', '/', testhandle)
	app.router.add_route('POST', '/feed', feedHandle)
	app.router.add_route('GET', '/stop', stopHandle)
	app.on_startup.append(init_feed)
	app.on_startup.append(init_ws)
	app.on_cleanup.append(cleanup_background_tasks)
	app.on_shutdown.append(on_shutdown)
	web.run_app(app, host=host, port=port)

if __name__ == '__main__':
	main()
