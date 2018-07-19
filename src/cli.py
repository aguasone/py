import os
import re
import scipy.misc
import warnings
import face_recognition.api as face_recognition
import time
import logging
import cv2
import numpy as np

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import dlib

# TODO Remove deleted encodings

#LOGS
logging.basicConfig(format="[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s")

logger = logging.getLogger('async')
logger.setLevel(logging.DEBUG)

local = {}
local['detector'] = dlib.get_frontal_face_detector()
local['predictor'] = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
local['fa'] = FaceAligner(local['predictor'], desiredFaceWidth=256)

class Recon():
	def __init__(self, known_people_folder, tolerance=0.6, show_distance=True):
		self.tolerance = tolerance
		self.show_distance = show_distance
		self.known_people_folder =  known_people_folder
		self.known_names = []
		self.known_face_encodings = []
		self.scan_known_people()


	def scan_known_people(self):
		self.known_names = []
		self.known_face_encodings = []
		for file in self.image_files_in_folder():
			basename = os.path.splitext(os.path.basename(file))[0]
			img = face_recognition.load_image_file(file)

			encodings = face_recognition.face_encodings(img)
				# if len(encodings) > 1:
				# 	print("WARNING: More than one face found in {}. Only considering the first face.".format(file))

				# if len(encodings) == 0:
				# 	print("WARNING: No faces found in {}. Ignoring file.".format(file))
				# else:
			self.known_names.append(basename)
			self.known_face_encodings.append(encodings[0])
		logger.debug("loaded: {}".format(self.known_names))

	async def test_image(self, image_to_check):
		unknown_image = image_to_check
		name = ['0.unknown']
		distances = []
		result = []

		if (self.known_names):
			# Scale down image if it's giant so things run a little faster
			if unknown_image.shape[1] > 1600:
				scale_factor = 1600.0 / unknown_image.shape[1]
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					unknown_image = scipy.misc.imresize(unknown_image, scale_factor)

			unknown_encodings = face_recognition.face_encodings(unknown_image)

			for unknown_encoding in unknown_encodings:
				distances = face_recognition.face_distance(self.known_face_encodings, unknown_encoding)
				result = list(distances <= self.tolerance)
				dist = None

				if True in result:
					name = [name for is_match, name, distances in zip(result, self.known_names, distances) if is_match]
					dist = [distances for is_match, name, distances in zip(result, self.known_names, distances) if is_match]
					#print ([(name) for is_match, name, distances in zip(result, self.known_names, distances) if is_match])
					#name = match
				else:
					name = ['0.unknown']

				if name != ['0.unknown']:
					names = name
					logger.debug("recognized: {}".format(len(names)))
					#logger.debug("name: {}".format(names))
					dist_np = np.array(dist)
					#logger.debug("distances: {}".format(dist_np))

					distances = []
					name = []
					distances.append(dist_np[dist_np.argmin()])
					name.append(names[dist_np.argmin()])

						# for i,v in enumerate(name):
						# 	if not v.startswith( '0.unknown' ):
						# 		name = []
						# 		name.append(v)
						# 		break
					return name, distances

		return name, distances

	async def unknown_people(self, img, name, photo):
		if photo is not None:
			print ('upeople')
			img = photo

		encodings = face_recognition.face_encodings(img)

		if len(encodings) != 0:
			self.known_names.append(name)
			self.known_face_encodings.append(encodings[0])

		#logger.debug("unknown: {}".format(self.known_names))
		logger.debug("unknown #: {}".format(len(self.known_names)))

		return len(encodings)

	async def delete_unknown_names(self, name):
		for i,v in enumerate(self.known_names):
			if v == name:
				try:
					del self.known_names[i]
					del self.known_face_encodings[i]
					logger.debug("unknown # after delete: {}".format(len(self.known_names)))
				except Exception as e:
					print ("Already Deleted: ", e)

	def image_files_in_folder(self):
		return [os.path.join(self.known_people_folder, f) for f in os.listdir(self.known_people_folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]
