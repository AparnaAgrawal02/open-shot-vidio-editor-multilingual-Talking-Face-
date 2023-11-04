from flask import Flask, request, send_file
from flask_ngrok import run_with_ngrok
import subprocess, os
import random, string
import sys


app = Flask(__name__)
#run_with_ngrok(app)
@app.route("/")
def index():
	return "Welcome to tool page"

@app.route("/filler_image/<filename>")
def filler_image(filename):
	return send_file(os.getcwd()+'/filler/' + filename)


@app.route("/filler/<name>", methods=["POST"])
def filler(name):
	f = request.files['file']
	filename = os.path.join('received', f.filename)
	f.save(filename)
	
	# outfile = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))
	outfile = os.path.join('filler', name)
	
	cmd = 'python filler/filler_selection.py {} {}'.format(filename, outfile)
	print(cmd)
	subprocess.call(cmd, shell=True)

	avi = os.getcwd()+"/filler/{}.avi".format(name)
	mp4 = os.getcwd()+"/filler/{}.mp4".format(name)
	cmd = 'ffmpeg -i {} {}'.format(avi, mp4)
	subprocess.call(cmd, shell=True)

	return send_file(mp4)

@app.route("/tts", methods=["POST"])
def tts():
	f = request.files['file']
	filename = os.path.join('received', f.filename)
	f.save(filename)

	with open(filename, 'r') as f:
		text = f.readlines()[0].strip()
	
	outfile = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7))) + '.wav'
	outfile = os.path.join('tts_out', outfile)
	
	cmd = 'python glow-tts/inference.py --text "{}" --outfile {}'.format(text, outfile)
	print(cmd)
	subprocess.call(cmd, shell=True)

	return send_file(outfile)

@app.route("/tts1", methods=["POST"])
def tts1():
	f = request.files['file']
	filename = os.path.join('received', f.filename)
	f.save(filename)

	with open(filename, 'r') as f:
		text = f.readlines()[0].strip()
	
	outfile = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7))) + '.wav'
	outfile = os.path.join('tts_out', outfile)
	
	cmd = 'python glow-tts/inference.py --ckpt pretrained.pth --text "{}" --outfile {}'.format(text, outfile)
	print(cmd)
	subprocess.call(cmd, shell=True)

	return send_file(outfile)	

@app.route("/tts2", methods=["POST"])
def tts2():
	f = request.files['file']
	filename = os.path.join('received', f.filename)
	f.save(filename)

	with open(filename, 'r') as f:
		text = f.readlines()[0].strip()
	
	outfile = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7))) + '.wav'
	outfile = os.path.join('tts_out', outfile)
	
	cmd = 'python Googletts/googletts.py --text "{}" --outfile {}'.format(text, outfile)
	print(cmd)
	subprocess.call(cmd, shell=True)

	return send_file(outfile)	



@app.route('/test', methods=["POST"])
def test():
	# f1 = request.files['file']
	f2 = request.data
	# print(f1.filename, f2)
	print(f2)
	d = f2.decode('ascii')
	print(d)
	return "done!"

@app.route("/lipsync/<fps>", methods=["POST"])
def lipsync(fps):
	f1 = request.files['file1']
	f2 = request.files['file2']
	
	box = None
	try:
		f3 = request.files['file3']
		filename = os.path.join('received', f3.filename)
		f3.save(filename)
		with open(filename, 'r') as f:
			text = f.readlines()[0].strip()
		box = text
	except:
		print("No file3 found")

	filename1 = os.path.join('received', f1.filename)
	filename2 = os.path.join('received', f2.filename)
	
	f1.save(filename1)
	f2.save(filename2)
	
	outfile = f1.filename.split('.')[0]
	temp = os.path.join('ls_out', 'temp', outfile + '.avi')
	res = os.path.join('ls_out', 'res', outfile + '.mp4')
	# Wav2Lip/models/wav2lip.pth
	cmd = 'python Wav2Lip/inference2.py \
			--checkpoint_path Wav2Lip/checkpoints/wav2lip.pth \
			--face {} \
			--audio {} \
			--fps {} \
			--tempfile {} \
			--outfile {}'.format(filename1, filename2, fps, temp, res)
                       #--checkpoint_path Wav2Lip/checkpoints/wav2lipgan_hd_288_192_constrained_10f_658k.pth 
	if box is not None:
		cmd = cmd + " --boxsearch {} --landmarks True".format(box)
		# cmd = cmd + " --boxsearch {}".format(box)
	print('#################################', cmd, flush=True)
	subprocess.call(cmd, shell=True)

	return send_file(res)	
	# return "lipsync"

@app.route("/fomm", methods=["POST"])
def fomm():
	f1 = request.files['file1']
	f2 = request.files['file2']

	filename1 = os.path.join(sys.path[0], 'received', f1.filename)
	filename2 = os.path.join(sys.path[0], 'received', f2.filename)
	
	f1.save(filename1)
	f2.save(filename2)
	
	outfile = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))

	res = os.path.join(sys.path[0], 'ls_out', 'res', outfile + '.mp4')
	cmd = 'python first-order-model/demo.py \
			--source_image {} \
			--driving_video {} \
			--result_video {}'.format(filename1, filename2, res)

	print(cmd)
	subprocess.call(cmd, shell=True)

	return send_file(res)

@app.route("/mit", methods=["POST"])
def mit():
	print("IN MIT")
	f1 = request.files['file1']
	f2 = request.files['file2']

	filename1 = os.path.join('received', f1.filename)
	filename2 = os.path.join('received', f2.filename)
	
	f1.save(filename1)
	f2.save(filename2)
	
	outfile = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))

	resavi = os.path.join('ls_out', 'res', outfile + '.avi')
	res = os.path.join('ls_out', 'res', outfile + '.mp4')
	cmd = 'python MakeItTalk/inf.py \
			--img_path {} \
			--audio_path {} \
			--outfile1 {} \
			--outfile2 {}'.format(filename1, filename2, resavi, res)

	print(cmd)
	subprocess.call(cmd, shell=True)

	return send_file(res)

@app.route("/slide_parser1", methods=["POST"])
def slide_parser1():
	f1 = request.files['file1']

	filename1 = os.path.join('received', f1.filename)
	
	f1.save(filename1)
	
	outfile = f1.filename.split('.')[0]

	res = os.path.join('ls_out', 'res', outfile + '.txt')

	cmd = 'python LayoutParser/get_text.py \
			--filepath {} \
			--outfile {}'.format(filename1, res)

	print(cmd)
	subprocess.call(cmd, shell=True)

	return send_file(res)

@app.route("/speechtotext", methods=["POST"])
def speechtotext():
	f1 = request.files['file1']

	filename1 = os.path.join('received', f1.filename)
	
	f1.save(filename1)
	
	outfile = f1.filename.split('.')[0]

	res = os.path.join('stt_out', 'res', outfile + '.txt')

	cmd = 'python GspeechToText/speechToText.py \
			--audio {} \
			--outfile {}'.format(filename1, res)

	print(cmd)
	subprocess.call(cmd, shell=True)

	return send_file(res)
	
@app.route("/translate/<lang>", methods=["POST"])
def translate(lang):
	f1 = request.files['file']

	filename1 = os.path.join('received', f1.filename)
	
	f1.save(filename1)
	
	outfile = f1.filename.split('.')[0]

	res = os.path.join('stt_out', 'res', outfile + 'a.txt')

	cmd = 'python GTranslate/translate.py \
			--text {} --lang {} \
			--outfile {}'.format(filename1,lang, res)

	print(cmd)
	subprocess.call(cmd, shell=True)

	return send_file(res)
	
	

@app.route("/slide_parser2/<filename>/<fps>", methods=["POST"])
def slide_parser2(filename, fps):
	f1 = request.files['file1']

	# text file
	filename1 = os.path.join('received', f1.filename)
	
	f1.save(filename1)
	
	outfile = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))

	video_file = os.path.join('received', filename + '.mp4')
	pkl_file = os.path.join('ls_out', 'res', filename + '.pkl')
	# textfile = os.path.join('ls_out', 'res', filename + '.txt')

	res = os.path.join('ls_out', 'res', outfile + '.avi')
	cmd = 'python LayoutParser/apply_text.py \
			--filepath {} \
			--outfile {} \
			--fps {} \
			--pkl {} \
			--textfile {}'.format(video_file, res, int(fps), pkl_file, filename1)

	print(cmd)
	subprocess.call(cmd, shell=True)

	return send_file(res)

@app.route("/upload", methods=["POST"])
def upload():
	f = request.data
	print(f)
	# f = request.files['file']
	# f.save(f.filename)
	return "file uploaded"

@app.route("/download/<file>", methods=["GET"])
def download(file):
	fl = os.path.join('ls_out', 'res', file)
	if not os.path.isfile(fl):
		return "File does not exist"
	return send_file(fl)

if __name__ == "__main__":
	app.run(port=3000, debug=True)
	#app.run(debug=True)
