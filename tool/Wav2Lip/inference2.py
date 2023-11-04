from os import listdir, path
import numpy as np
from pyrsistent import m
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection

import time
from models import Wav2Lip
#from models import Wav2Lip_large_window_hd_new_10f_large as Wav2Lip
import platform
import face_alignment
out = None

BASE_DIR = os.environ['PWD']


parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='results/result_voice.mp4')
parser.add_argument('--tempfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='temp/result.avi')
parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=16)
# parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=32)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')


parser.add_argument('--boxsearch', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('--landmarks', default=False, type=bool, help="replace portion inside face landmarks")

args = parser.parse_args()
args.img_size = (96, 96)
#args.img_size = (192, 288)

args.face = os.path.join(BASE_DIR, args.face)
args.audio = os.path.join(BASE_DIR, args.audio)
args.outfile = os.path.join(BASE_DIR, args.outfile)
args.tempfile = os.path.join(BASE_DIR, args.tempfile)

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
	try:
		gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		face_coordinates = trained_face_data.detectMultiScale(gray_img)
		# print(face_coordinates)
		for coordinate in face_coordinates:
			(x, y, w, h) = coordinate
			colors = np.random.randint(1, 255, 3)
		#     cv2.rectangle(image, (x, y), (x + w, y + h), (int(colors[0]), int(colors[1]), int(colors[2])), thickness=2)
		# cv2.imshow('Image', image)
		# cv2.waitKey(0)

		return [[x, y, x+w, y+h]]
	except:
		return [[-1, -1, -1, -1]]

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)
	#detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
	batch_size = args.face_det_batch_size
	images = np.array(images)
	# temp_images = np.copy(images)
	# temp_images[:, :, temp_images.shape[2]//2:] = 0
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	# predictions = []
	# for i in tqdm(range(0, len(images))):
	# 	predictions.extend(detect_faces(images[i]))

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
		# 	cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
		# 	raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
		# if rect[0] == -1:
			x1, y1, x2, y2 = [-1, -1, -1, -1]
		else:
			# print("PRINTINGPRINTING", rect)
			y1 = max(0, rect[1] - pady1)
			y2 = min(image.shape[0], rect[3] + pady2)
			x1 = max(0, rect[0] - padx1)
			x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	# if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	# results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
	results = []
	for image, (x1, y1, x2, y2) in zip(images, boxes):
		if x1 == -1:
			results.append([image, (y1, y2, x1, x2)])
		else:
			results.append([image[y1: y2, x1:x2], (y1, y2, x1, x2)])

	del detector
	return results 

def datagen(full_video_frames, frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch, full_video_frames_batch = [], [], [], [], []

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		# face = cv2.resize(face, (args.img_size, args.img_size))
		face = cv2.resize(face, args.img_size)
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		full_video_frames_batch.append(full_video_frames[idx].copy())
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			# img_masked[:, args.img_size//2:] = 0
			img_masked[:, args.img_size[1]//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch, full_video_frames_batch
			img_batch, mel_batch, frame_batch, coords_batch, full_video_frames_batch = [], [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		# img_masked[:, args.img_size//2:] = 0
		img_masked[:, args.img_size[1]//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch, full_video_frames_batch

mel_step_size = 16
#mel_step_size = 28
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def get_mel_chunks(full_frames,mel_chunks_used,still_reading):
	fps = args.fps
	print ("Number of frames available for inference: "+str(len(full_frames)))
	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		pth = '/home/f2f_translation/tool/Wav2Lip/temp/temp.wav'
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, pth)

		subprocess.call(command, shell=True)
		args.audio = pth

	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	# mel_idx_multiplier = 60./fps 
	mel_idx_multiplier = 80./fps 
	# mel_idx_multiplier = 100./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	# changed
	mel_chunks = np.array(mel_chunks)
	#print(mel_chunks)
	return mel_chunks

def wave_to_lip_inf(full_frames,mel_chunks_used,still_reading,full_video_frames,out,mel_chunks_actual):
	if out==None:
		mel_chunks_actual = get_mel_chunks(full_frames,mel_chunks_used,still_reading)
	
	mel_chunks = mel_chunks_actual[mel_chunks_used:]
	diff = len(full_frames) - mel_chunks.shape[0]
	if diff > 0:
		mel_chunks = np.concatenate((mel_chunks, np.zeros((diff, mel_chunks.shape[1], mel_chunks.shape[2]))))
	elif still_reading:
		mel_chunks=mel_chunks[:len(full_frames)]
	

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size
	gen = datagen(full_video_frames.copy(), full_frames.copy(), mel_chunks)

	# haarcascade_clf = "/home/anchit.gupta/FLASK/facial-landmark-detection/data/haarcascade_frontalface_alt2.xml"
	# face_detector = cv2.CascadeClassifier(haarcascade_clf)
	face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

	for i, (img_batch, mel_batch, frames, coords, full_frames_video) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			model = load_model(args.checkpoint_path)
			print ("Model loaded")

			frame_h, frame_w = full_video_frames[0].shape[:-1]
			if(out == None):
				print(out)
				out = cv2.VideoWriter(args.tempfile, 
									cv2.VideoWriter_fourcc(*'DIVX'), args.fps, (frame_w, frame_h))

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
		print(img_batch.shape,mel_batch.shape)
		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		idx = 0
		for p, f, c, F in zip(pred, frames, coords, full_frames_video):
			y1, y2, x1, x2 = c

			# print(c)
			if y1 == -1:
				out.write(F)
				print("writing same frame")
			else:
				p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

				# facial landmark code
				# print(x1, x2, y1, y2)
				if args.landmarks:
					try:
						pred = face_detector.get_landmarks(f[y1:y2, x1:x2])[0]

						# print(pred)
						pts = []
						for i, (x,y) in enumerate(pred):
							if i > 26:
								break
							# print(x, y)
							pts.append([int(x), int(y)])
							# cv2.circle(f, (int(x), int(y)), 1, (255, 0, 0), 2)

						pts = np.array(pts)
						# print(pts)
						pts[17:] = pts[17:][::-1]
						rect = cv2.boundingRect(pts)
						x, y, w, h = rect
						# print(rect, p.shape)
						cropped = np.copy(p)
						# pts = pts - pts.min(axis=0)
						mask = np.zeros(cropped.shape[:2], np.uint8)
						cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
						dst = cv2.bitwise_and(cropped, cropped, mask=mask)
						bg = np.ones_like(cropped, np.uint8)*255
						cv2.bitwise_not(bg,bg, mask=mask)
						dst2 = bg + dst

						frame_copy = np.copy(f).astype(np.float)
						frame_copy[y1:y2, x1:x2][:, :, 0] = frame_copy[y1:y2, x1:x2][:, :, 0] - mask
						frame_copy[y1:y2, x1:x2][:, :, 1] = frame_copy[y1:y2, x1:x2][:, :, 1] - mask
						frame_copy[y1:y2, x1:x2][:, :, 2] = frame_copy[y1:y2, x1:x2][:, :, 2] - mask
						frame_copy[frame_copy < 0] = 0.0

						frame_copy[y1:y2, x1:x2] = frame_copy[y1:y2, x1:x2] + dst
						cv2.imwrite('faces/mask_{:05d}.png'.format(idx), mask)
						cv2.imwrite('faces/dst_{:05d}.png'.format(idx), dst)
						cv2.imwrite('faces/dst2_{:05d}.png'.format(idx), dst2)
						cv2.imwrite('faces/frame_copy_{:05d}.png'.format(idx), frame_copy)
						cv2.imwrite('faces/cropped_{:05d}.png'.format(idx), cropped)
						cv2.imwrite('faces/f_{:05d}.png'.format(idx), f)
						idx += 1
						# print('The y coordinates are: ',y1,'  ',y2)
						# frame_copy = np.array(frame_copy, dtype=np.uint8)
						y1_up = max(0,y1-6)
						y2_down = min(frame_copy.shape[1],y1+6)
						x1_left = max(0,x1-6)
						x2_right = min(frame_copy.shape[0],x2+6)


						# frame_copy[y1_up:y2_down,x1_left:x2_right] = cv2.medianBlur(frame_copy[y1_up:y2_down,x1_left:x2_right].astype(np.uint8),7)
						fcc = cv2.medianBlur(frame_copy.astype(np.uint8),5)
						# fcc = 
						# f = frame_copy
						f = fcc

					except:
						print("no landmarks detected")
						cv2.imwrite('faces/aaaaaa_{:05d}.png'.format(idx), p)
						f[y1:y2, x1:x2] = p
				else:
					f[y1:y2, x1:x2] = p

				if args.boxsearch[0] != -1:
					y1, y2, x1, x2 = args.boxsearch
					F[y1:y2, x1:x2] = f
					out.write(F)
				else:	
					out.write(f)
	return [out,mel_chunks_actual]

def main():

	curr = time.time()
	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')

	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(args.face)]
		fps = args.fps

	else:
		video_stream = cv2.VideoCapture(args.face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)
		print('Reading video frames...')

		full_frames = []
		full_video_frames = []
		mel_chunks_used = 0
		x=0
		out = None
		mel_chunks=None
		landmark_detector  = cv2.face.createFacemarkLBF()
		landmark_detector.loadModel(os.path.join(BASE_DIR, 'Wav2Lip/LFBmodel.yaml'))
		while 1:
			still_reading, frame = video_stream.read()

			if len(full_frames)==1000 or not still_reading :
				[out,mel_chunks] = wave_to_lip_inf(full_frames,mel_chunks_used,still_reading,full_video_frames,out,mel_chunks)
				print(out)
				mel_chunks_used+=len(full_frames)
				full_frames = []
				full_video_frames = []



				if not still_reading:
					video_stream.release()
					break
				x+=1
				print(x)
				
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

			if args.rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			full_video_frames.append(frame[y1:y2, x1:x2])
			if args.boxsearch[0] != -1:
				y1, y2, x1, x2 = args.boxsearch

			frame = frame[y1:y2, x1:x2]
			full_frames.append(frame)
			
				

	if out!=None:
		out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, args.tempfile, args.outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')
	print(time.time()-curr)
if __name__ == '__main__':
	main()
