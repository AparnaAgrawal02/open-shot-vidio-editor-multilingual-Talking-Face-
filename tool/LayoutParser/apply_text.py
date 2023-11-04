import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import argparse
from tqdm import tqdm
import sys
import pickle as pkl

parser = argparse.ArgumentParser(description='Run OCR on slides')
parser.add_argument('--filepath', default='', type=str, help="video")
parser.add_argument('--outfile', default='', type=str, help="output file path")
parser.add_argument('--fps', default=25, type=int, help="fps")
parser.add_argument('--font_size', default=None, help="font size")
parser.add_argument('--pkl', default='', type=str, help="output file path")
parser.add_argument('--textfile', default='', type=str, help="output file path")


args = parser.parse_args()

class ApplyText:

	def __init__(self):
		self.text_in_a_line = []
		self.coords = []
		self.hw = [] # height width
		self.translated_text = []
		self.frames = []
		self.outfile = args.outfile
		self.file_path = args.filepath
		self.fps = args.fps

	def getBgColor(self, img):
		
		zero = np.bincount(img[:, :, 0].reshape(-1))
		one = np.bincount(img[:, :, 1].reshape(-1))
		two = np.bincount(img[:, :, 2].reshape(-1))
		b = np.argmax(zero)
		g = np.argmax(one)
		r = np.argmax(two)

		return (b, g, r)

	def getFontColor(self, img):
		
		zero = np.bincount(img[:, :, 0].reshape(-1))
		one = np.bincount(img[:, :, 1].reshape(-1))
		two = np.bincount(img[:, :, 2].reshape(-1))
		b = np.argmax(zero)
		g = np.argmax(one)
		r = np.argmax(two)

		zero[b] = -1
		one[g] = -1
		two[r] = -1

		b2 = np.argmax(zero)
		g2 = np.argmax(one)
		r2 = np.argmax(two)

		return (r2, g2, b2)

	def writeTextonImage(self, img):

		PILimage = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

		if args.font_size == None:
			args.font_size = int(0.4 * (self.coords[1][0][1] - self.coords[0][0][1]))
		# print(sys.path[0])
		title_font = ImageFont.truetype('{}/Hind-Regular.ttf'.format(sys.path[0]), args.font_size)
		
		# font_color = self.getFontColor(img) # black
		font_color = (0, 0, 0) # black
		for i in range(len(self.translated_text)):
			title_text = self.translated_text[i].strip()
			image_editable = ImageDraw.Draw(PILimage)
			image_editable.text(self.coords[i][0], title_text, font_color, font=title_font)

		numpyImg = cv2.cvtColor(np.array(PILimage), cv2.COLOR_RGB2BGR)
		return numpyImg

	def writeVideo(self):

		outImgs = []
		self.bgr = self.getBgColor(self.frames[0])
		for i, img in tqdm(enumerate(self.frames), total=len(self.frames)):
			for j in range(len(self.coords)):
				for k in range(len(self.coords[j])):
					x1, y1, x2, y2 = self.coords[j][k]
					img[y1-5:y2+5, x1-5:x2+5] = self.bgr
			out = self.writeTextonImage(img)
			outImgs.append(out)

		frame_h, frame_w = outImgs[0].shape[:-1]
		vid_writer = cv2.VideoWriter(self.outfile, 
								cv2.VideoWriter_fourcc(*'DIVX'), self.fps, (frame_w, frame_h))

		for img in tqdm(outImgs):
			vid_writer.write(img)
		vid_writer.release()
		
	def main(self):

		video_stream = cv2.VideoCapture(self.file_path)
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break

			self.frames.append(frame)

		print("Read pkl file")
		with open(args.pkl, 'rb') as f:
			self.coords = pkl.load(f)
		f.close()

		print("Read text file")
		with open(args.textfile, 'r') as t:
			self.translated_text = t.readlines()
		t.close()

		# get translated text
		print("Start writeVideo()")
		self.writeVideo()


gt = ApplyText()
gt.main()