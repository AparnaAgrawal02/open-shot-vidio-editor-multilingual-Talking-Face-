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
translator = from_pretrained(tag='mm-all-iter1')

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
parser.add_argument('--vid', default=False, type=bool, help="video")
parser.add_argument('--file_path', default='', type=str, help="video")
parser.add_argument('--outfile', default='', type=str, help="output file path")
parser.add_argument('--fps', default=25, type=int, help="fps")
parser.add_argument('--tgt_lang', default='hi', type=str, help="target translation lang")
parser.add_argument('--src_lang', default='en', type=str, help="source translation lang")
parser.add_argument('--same', default=False, type=bool, help="if slide is same for all duration")

args = parser.parse_args()

def getBgColor(img):
	
	r = np.argmax(np.bincount(img[:, :, 0].reshape(-1)))
	b = np.argmax(np.bincount(img[:, :, 1].reshape(-1)))
	g = np.argmax(np.bincount(img[:, :, 2].reshape(-1)))
	return (r, b, g)

def ocrOnImage(img):

	data = pytesseract.image_to_data(img, output_type=Output.DICT)
	# print(data)
	n_boxes = len(data['level'])

	text_in_a_line, coords = [], []

	rbg = getBgColor(img)
	H, idx = 0, -1

	for i in range(n_boxes):
		(x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
		text = data['text'][i]
		for sym in '.(),:?!':
			text = text.replace(sym, '')

		if text.isalnum():
			# print(data['text'][i], x, x+w, y, y + h)
			if -10 <= H-y <= 10:
				text_in_a_line[idx].append(data['text'][i])
				coords[idx].append((x, y))
			else:
				idx += 1
				text_in_a_line.append([data['text'][i]])
				coords.append([(x, y)])
			H = y
			img[y:y+h, x:x+w] = rbg
			# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
			# img[y:y+h, x:x+w][:, :, 0] = a
			# img[y:y+h, x:x+w][:, :, 1] = b
			# img[y:y+h, x:x+w][:, :, 2] = c
			
	# cv2.imshow('img', img)
	# cv2.waitKey(0)
	# print(text_in_a_line)
	translated_text = []

	for i, txt in enumerate(text_in_a_line):
		
		text = " ".join(txt)
		dst_txt = translator(text, tgt_lang=args.tgt_lang, src_lang=args.src_lang)

		t = ""
		for tr in dst_txt:
			t += tr['tgt'] + " "
		translated_text.append(t)

	# print(translated_text)
	PILimage = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	title_font = ImageFont.truetype('{}/hindi/Hind-Regular.ttf'.format(sys.path[0]), 50)

	font_color = (0, 0, 0) # black
	for i in range(len(translated_text)):
		title_text = translated_text[i]
		image_editable = ImageDraw.Draw(PILimage)
		image_editable.text(coords[i][0], title_text, font_color, font=title_font)

	numpyImg = cv2.cvtColor(np.array(PILimage), cv2.COLOR_RGB2BGR)
	return numpyImg

def main():

	if args.vid:
		video_stream = cv2.VideoCapture(args.file_path)
		frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break

			frames.append(frame)
	else:
		frames = [cv2.imread(args.file_path)]

	outImgs = []
	if args.same:
		out = ocrOnImage(frames[0])
		for f in tqdm(frames):
			outImgs.append(out)
	else:
		for f in tqdm(frames):
			out = ocrOnImage(f)
			outImgs.append(out)
			# cv2.imshow('img', out)
			# cv2.waitKey(0)

	if args.vid:
		frame_h, frame_w = outImgs[0].shape[:-1]
		vid_writer = cv2.VideoWriter(args.outfile, 
								cv2.VideoWriter_fourcc(*'DIVX'), args.fps, (frame_w, frame_h))

		for img in tqdm(outImgs):
			vid_writer.write(img)

		vid_writer.release()
	else:
		cv2.imwrite(args.outfile, outImgs[0])

if __name__ == '__main__':
	main()