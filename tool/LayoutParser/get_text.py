import cv2
from ilmulti.translator import from_pretrained
import numpy as np
import argparse
import sys
import layoutparser as lp
from googletrans import Translator
translator = Translator(service_urls=['translate.googleapis.com'])

import pickle as pkl
parser = argparse.ArgumentParser(description='Run OCR on slides')
parser.add_argument('--filepath', default='', type=str, help="video")
parser.add_argument('--outfile', default='', type=str, help="output file path")
parser.add_argument('--tgt_lang', default='hi', type=str, help="target translation lang")
parser.add_argument('--src_lang', default='en', type=str, help="source translation lang")
parser.add_argument('--src_lang_slide', default='eng', type=str, help="slide language")
args = parser.parse_args()

class GetText:

	def __init__(self):
		#self.translator = from_pretrained(tag='mm-all-iter1')
		self.translator = Translator(service_urls=['translate.googleapis.com'])
		self.text_in_a_line = []
		self.coords = []
		self.hw = [] # height width
		self.translated_text = []
		self.frames = []
		self.outfile = args.outfile
		self.file_path = args.filepath
		self.src_lang = args.src_lang
		self.tgt_lang = args.tgt_lang
		self.src_lang_slide = args.src_lang_slide

	def getBgColor(self, img):
		
		r = np.argmax(np.bincount(img[:, :, 0].reshape(-1)))
		b = np.argmax(np.bincount(img[:, :, 1].reshape(-1)))
		g = np.argmax(np.bincount(img[:, :, 2].reshape(-1)))
		return (r, b, g)

	def ocrOnImage(self, img):

		img = img[..., ::-1]
		#cv2.imshow(img)
		ocr_agent = lp.TesseractAgent(languages='eng')
		text = ocr_agent.detect(img, return_response=True)
		print(text)
		# get text with more than 50% confidence level
		# print(text)
		data = text["data"]
		x = [x for x in list(text["data"]["text"]) if str(x) != 'nan']
		print(" ".join(x),"hi")
                
		data_conf = data[data["conf"] > 50]
		text["data"] = data_conf
		layout = ocr_agent.gather_data(text, agg_level=lp.TesseractFeatureType.WORD)

		txt = " ".join(x)
		# print(txt)
		for i in text["text"].split('\n'):
			# print(i)
			if i in txt and i != " " and i != "":
				self.text_in_a_line.append(i)
		# print(self.text_in_a_line)
		j = 0
		for i in layout:
			if i.text != "" and i.text != " ":
				x1, x2, y1, y2 = i.block.x_1, i.block.x_2, i.block.y_1, i.block.y_2
				if j < len(self.text_in_a_line) and i.text == self.text_in_a_line[j].split(" ")[0] :
					self.coords.append([[x1, y1, x2, y2]])
					j += 1
				else:
					# print(i.text, self.text_in_a_line[j].split(" ")[0])
					print(i, j,"life")
					self.coords[j-1].append([x1, y1, x2, y2])
		

		for i, txt in enumerate(self.text_in_a_line):
			
			#dst_txt = self.translator(txt, tgt_lang=self.tgt_lang, src_lang=self.src_lang)
			dst_txt = translator.translate(txt, dest=self.tgt_lang)

			t = ""
			#for tr in dst_txt:
			#	t += tr['tgt'] + " "
			t = dst_txt.text
			t = t.encode(encoding = 'UTF-8', errors = 'strict').decode()
			self.translated_text.append(t)

		combined_text = '\n'.join(self.translated_text)
		combined_text = combined_text.encode(encoding = 'UTF-8', errors = 'strict').decode()
		f = open(self.outfile, 'w')
		# f.write(self.file_path)
		f.write(combined_text)
		f.close()

		name = self.outfile.split('.')
		coordinates_file = name[0] + '.pkl'
		with open(coordinates_file, 'wb') as cf:
			pkl.dump(self.coords, cf)
		cf.close()

	def main(self):

		video_stream = cv2.VideoCapture(self.file_path)
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break

			self.frames.append(frame)

		# get translated text
		self.ocrOnImage(self.frames[0])


gt = GetText()
gt.main()
