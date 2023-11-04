import pytesseract
from pytesseract import Output
import cv2
from ilmulti.translator import from_pretrained
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import argparse
from tqdm import tqdm
import sys

# print(sys.path[0])
# translator = from_pretrained(tag='mm-all-iter1')

# parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
# parser.add_argument('--vid', default=False, type=bool, help="video")
# parser.add_argument('--file_path', default='', type=str, help="video")
# parser.add_argument('--outfile', default='', type=str, help="output file path")
# parser.add_argument('--fps', default=25, type=int, help="fps")
# parser.add_argument('--tgt_lang', default='hi', type=str, help="target translation lang")
# parser.add_argument('--src_lang', default='en', type=str, help="source translation lang")
# parser.add_argument('--same', default=False, type=bool, help="if slide is same for all duration")

# args = parser.parse_args()

class SlidesOCR:

	def __init__(self):
		self.translator = from_pretrained(tag='mm-all-iter1')
		self.text_in_a_line = []
		self.coords = []
		self.hw = [] # height width
		self.translated_text = []
		self.frames = []
		self.outfile = ""
		self.file_path = ""
		self.src_lang = 'en'
		self.tgt_lang = 'hi'

	def getBgColor(self, img):
		
		r = np.argmax(np.bincount(img[:, :, 0].reshape(-1)))
		b = np.argmax(np.bincount(img[:, :, 1].reshape(-1)))
		g = np.argmax(np.bincount(img[:, :, 2].reshape(-1)))
		return (r, b, g)

	def writeTextonImage(self, img, font_size=None):

		PILimage = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

		if font_size == None:
			font_size = int(0.5 * (self.coords[1][0][1] - self.coords[0][0][1]))
		# print(sys.path[0])
		title_font = ImageFont.truetype('{}/slides_ocr/hindi/Hind-Regular.ttf'.format(sys.path[0]), font_size)
		
		font_color = (0, 0, 0) # black
		for i in range(len(self.translated_text)):
			title_text = self.translated_text[i]
			image_editable = ImageDraw.Draw(PILimage)
			image_editable.text(self.coords[i][0], title_text, font_color, font=title_font)

		numpyImg = cv2.cvtColor(np.array(PILimage), cv2.COLOR_RGB2BGR)
		return numpyImg

	def writeVideo(self, font_size=None):

		outImgs = []
		for i, img in enumerate(self.frames):
			for j in range(len(self.coords)):
				for k in range(len(self.coords[j])):
					x, y = self.coords[j][k]
					w, h = self.hw[j][k]
					img[y:y+h, x:x+w] = self.rbg
			out = self.writeTextonImage(img, font_size)
			outImgs.append(out)

		frame_h, frame_w = outImgs[0].shape[:-1]
		vid_writer = cv2.VideoWriter(self.outfile, 
								cv2.VideoWriter_fourcc(*'DIVX'), self.fps, (frame_w, frame_h))

		for img in tqdm(outImgs):
			vid_writer.write(img)
		vid_writer.release()

	def textUpdate(self, textArray, font_size=None):
		self.translated_text = textArray
		self.writeVideo(font_size);


	def ocrOnImage(self, img):

		data = pytesseract.image_to_data(img, output_type=Output.DICT)
		# print(data)
		n_boxes = len(data['level'])

		self.rbg = self.getBgColor(img)
		H, idx = 0, -1

		for i in range(n_boxes):
			(x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
			text = data['text'][i]
			for sym in '.(),:?!':
				text = text.replace(sym, '')

			if text.isalnum():
				# print(data['text'][i], x, x+w, y, y + h)
				if -10 <= H-y <= 10:
					self.text_in_a_line[idx].append(data['text'][i])
					self.coords[idx].append((x, y))
					self.hw[idx].append([w, h])
				else:
					idx += 1
					self.text_in_a_line.append([data['text'][i]])
					self.coords.append([(x, y)])
					self.hw.append([[w, h]])
				H = y
				img[y:y+h, x:x+w] = self.rbg
				# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
				# img[y:y+h, x:x+w][:, :, 0] = a
				# img[y:y+h, x:x+w][:, :, 1] = b
				# img[y:y+h, x:x+w][:, :, 2] = c
				
		# cv2.imshow('img', img)
		# cv2.waitKey(0)
		# print(text_in_a_line)

		for i, txt in enumerate(self.text_in_a_line):
			
			text = " ".join(txt)
			dst_txt = self.translator(text, tgt_lang=self.tgt_lang, src_lang=self.src_lang)

			t = ""
			for tr in dst_txt:
				t += tr['tgt'] + " "

			t = t.encode(encoding = 'UTF-8', errors = 'strict').decode()
			self.translated_text.append(t)

	def main(self, file_path, outfile, fps, same=True):

		self.file_path = file_path
		self.outfile = outfile
		self.fps = fps
		video_stream = cv2.VideoCapture(file_path)
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break

			self.frames.append(frame)

		# get translated text
		self.ocrOnImage(self.frames[0])
		return self.translated_text

	if __name__ == '__main__':
		main()