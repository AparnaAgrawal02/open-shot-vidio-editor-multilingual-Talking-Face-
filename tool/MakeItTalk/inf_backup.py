import sys
import os, glob
BASE_DIR = os.environ['PWD']
WORK_DIR = sys.path[0]
sys.path.append(os.path.join(WORK_DIR, "thirdparty/AdaptiveWingLoss"))
import numpy as np
import cv2
import argparse
from src.approaches.train_image_translation import Image_translation_block
import torch
import pickle
import face_alignment
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
import shutil
import time
import util.utils as util
from scipy.signal import savgol_filter
from src.approaches.train_audio2landmark import Audio2landmark_model

EXAMPLE_DIR = os.path.join(WORK_DIR, 'examples')

parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, default='')
parser.add_argument('--close_input_face_mouth', default=False, action='store_true')

parser.add_argument('--load_AUTOVC_name', type=str, default=os.path.join(EXAMPLE_DIR, 'ckpt/ckpt_autovc.pth'))
parser.add_argument('--load_a2l_G_name', type=str, default=os.path.join(EXAMPLE_DIR, 'ckpt/ckpt_speaker_branch.pth'))
parser.add_argument('--load_a2l_C_name', type=str, default=os.path.join(EXAMPLE_DIR, 'ckpt/ckpt_content_branch.pth')) #ckpt_audio2landmark_c.pth')
parser.add_argument('--load_G_name', type=str, default=os.path.join(EXAMPLE_DIR, 'ckpt/ckpt_116_i2i_comb.pth')) #ckpt_image2image.pth') #ckpt_i2i_finetune_150.pth') #c

parser.add_argument('--amp_lip_x', type=float, default=2.)
parser.add_argument('--amp_lip_y', type=float, default=2.)
parser.add_argument('--amp_pos', type=float, default=0.7)
parser.add_argument('--reuse_train_emb_list', type=str, nargs='+', default=[]) #  ['iWeklsXc0H8']) #['45hn7-LXDX8']) #['E_kmpT-EfOg']) #'iWeklsXc0H8', '29k8RtSUjE0', '45hn7-LXDX8',
parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default=EXAMPLE_DIR)

parser.add_argument('--test_end2end', default=True, action='store_true')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--use_prior_net', default=True, action='store_true')
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--init_content_encoder', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--write', default=False, action='store_true')
parser.add_argument('--segment_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
parser.add_argument('--use_11spk_only', default=False, action='store_true')
parser.add_argument('-f')
parser.add_argument('--img_path', required=True, help='img pth')
parser.add_argument('--audio_path', required=True, help='img pth')
parser.add_argument('--outfile1', required=True, help='img pth')
parser.add_argument('--outfile2', required=True, help='img pth')
opt_parser = parser.parse_args()

opt_parser.img_path = os.path.join(BASE_DIR, opt_parser.img_path)
opt_parser.audio_path = os.path.join(BASE_DIR, opt_parser.audio_path)
opt_parser.outfile1 = os.path.join(BASE_DIR, opt_parser.outfile1)
opt_parser.outfile2 = os.path.join(BASE_DIR, opt_parser.outfile2)

class MakeItTalk():
	def __init__(self):
		self.no = True

	def generate_inference(self, image, audio):
		copy_img_pth = os.path.join(EXAMPLE_DIR, image.split('/')[-1])
		copy_audio_pth = os.path.join(EXAMPLE_DIR, audio.split('/')[-1])
		if not os.path.exists(copy_img_pth):
			shutil.copyfile(image,copy_img_pth)
			print('Image Copied')
		if not os.path.exists(copy_audio_pth):
			shutil.copyfile(audio,copy_audio_pth)
			print('Audio Copied')


		default_head_name = image.split('/')[-1]	# the image name (with no .jpg) to animate
		ADD_NAIVE_EYE = True                 		# whether add naive eye blink
		CLOSE_INPUT_FACE_MOUTH = False       		# if your image has an opened mouth, put this as True, else False
		AMP_LIP_SHAPE_X = 2.                 		# amplify the lip motion in horizontal direction
		AMP_LIP_SHAPE_Y = 2.                 		# amplify the lip motion in vertical direction
		AMP_HEAD_POSE_MOTION = 0.7           		# amplify the head pose motion (usually smaller than 1.0, put it to 0. for a static head pose)
		opt_parser.jpg = default_head_name

		img =cv2.imread(os.path.join(EXAMPLE_DIR, opt_parser.jpg))
		face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
		pred = face_detector.get_landmarks(img)[0]
		pts = []
		for i, (x,y) in enumerate(pred):
			if i > 26:
				break
			# print(x, y)
			pts.append([int(x), int(y)])
		pts = np.array(pts)
		pts[17:] = pts[17:][::-1]
		rect = cv2.boundingRect(pts)
		x, y, w, h = rect
		img = img[y - 50: y+h + 20, x - 20:x+w + 20, :]

		img_height = img.shape[0]
		img_width = img.shape[1]
		if img_height != img_width != 256:
			img = cv2.resize(img, (256, 256))
		predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)
		shapes = predictor.get_landmarks(img)
		if (not shapes or len(shapes) != 1):
			print('Cannot detect face landmarks. Exit.')
			exit(-1)
		shape_3d = shapes[0]

		if(opt_parser.close_input_face_mouth):
			util.close_input_face_mouth(shape_3d)

		shape_3d[48:, 0] = (shape_3d[48:, 0] - np.mean(shape_3d[48:, 0])) * 1.05 + np.mean(shape_3d[48:, 0]) # wider lips
		shape_3d[49:54, 1] += 0.           # thinner upper lip
		shape_3d[55:60, 1] -= 1.           # thinner lower lip
		shape_3d[[37,38,43,44], 1] -=2.    # larger eyes
		shape_3d[[40,41,46,47], 1] +=2.    # larger eyes

		shape_3d, scale, shift = util.norm_input_face(shape_3d)

		au_data = []
		au_emb = []
		ains = [audio.split('/')[-1]]
		# ains = glob.glob1('examples', '*.wav')
		ains = [item for item in ains if item is not 'tmp.wav']
		ains.sort()
		for ain in ains:
			os.system('ffmpeg -y -loglevel error -i {}/{} -ar 16000 {}/tmp.wav'.format(EXAMPLE_DIR,ain,EXAMPLE_DIR))
			shutil.copyfile('{}/tmp.wav'.format(EXAMPLE_DIR), '{}/{}'.format(EXAMPLE_DIR,ain))

			# au embedding
			from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
			me, ae = get_spk_emb('{}/{}'.format(EXAMPLE_DIR, ain))
			au_emb.append(me.reshape(-1))

			print('Processing audio file', ain)
			c = AutoVC_mel_Convertor(EXAMPLE_DIR)

			au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=os.path.join(EXAMPLE_DIR, ain),
				   autovc_model_path=opt_parser.load_AUTOVC_name)
			au_data += au_data_i
		if(os.path.isfile('{}/tmp.wav'.format(EXAMPLE_DIR))):
			os.remove('{}/tmp.wav'.format(EXAMPLE_DIR))

		# landmark fake placeholder
		fl_data = []
		rot_tran, rot_quat, anchor_t_shape = [], [], []
		for au, info in au_data:
			au_length = au.shape[0]
			fl = np.zeros(shape=(au_length, 68 * 3))
			fl_data.append((fl, info))
			rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
			rot_quat.append(np.zeros(shape=(au_length, 4)))
			anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

		if(os.path.exists(os.path.join(EXAMPLE_DIR, 'dump', 'random_val_fl.pickle'))):
			os.remove(os.path.join(EXAMPLE_DIR, 'dump', 'random_val_fl.pickle'))
		if(os.path.exists(os.path.join(EXAMPLE_DIR, 'dump', 'random_val_fl_interp.pickle'))):
			os.remove(os.path.join(EXAMPLE_DIR, 'dump', 'random_val_fl_interp.pickle'))
		if(os.path.exists(os.path.join(EXAMPLE_DIR, 'dump', 'random_val_au.pickle'))):
			os.remove(os.path.join(EXAMPLE_DIR, 'dump', 'random_val_au.pickle'))
		if (os.path.exists(os.path.join(EXAMPLE_DIR, 'dump', 'random_val_gaze.pickle'))):
			os.remove(os.path.join(EXAMPLE_DIR, 'dump', 'random_val_gaze.pickle'))

		with open(os.path.join(EXAMPLE_DIR, 'dump', 'random_val_fl.pickle'), 'wb') as fp:
			pickle.dump(fl_data, fp)
		with open(os.path.join(EXAMPLE_DIR, 'dump', 'random_val_au.pickle'), 'wb') as fp:
			pickle.dump(au_data, fp)
		with open(os.path.join(EXAMPLE_DIR, 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
			gaze = {'rot_trans':rot_tran, 'rot_quat':rot_quat, 'anchor_t_shape':anchor_t_shape}
			pickle.dump(gaze, fp)

		# !pwd
		model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)
		if(len(opt_parser.reuse_train_emb_list) == 0):
			model.test(au_emb=au_emb)
		else:
			model.test(au_emb=None)


		fls = glob.glob1(EXAMPLE_DIR, 'pred_fls_*.txt')
		fls.sort()

		for i in range(0,len(fls)):
			fl = np.loadtxt(os.path.join(EXAMPLE_DIR, fls[i])).reshape((-1, 68,3))
			fl[:, :, 0:2] = -fl[:, :, 0:2]
			fl[:, :, 0:2] = fl[:, :, 0:2] / scale - shift

			if (ADD_NAIVE_EYE):
				fl = util.add_naive_eye(fl)

			# additional smooth
			fl = fl.reshape((-1, 204))
			fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
			fl[:, 48*3:] = savgol_filter(fl[:, 48*3:], 5, 3, axis=0)
			fl = fl.reshape((-1, 68, 3))

			''' STEP 6: Imag2image translation '''
			model = Image_translation_block(opt_parser, single_test=True)
			with torch.no_grad():
				# print("this", fls[i])
				model.single_test(outfile1=opt_parser.outfile1, outfile2=opt_parser.outfile2, jpg=img, fls=fl, filename=audio, prefix=opt_parser.jpg.split('.')[0],img_height=img_height,img_width=img_width)
				print('finish image2image gen')
			os.remove(os.path.join(EXAMPLE_DIR, fls[i]))

### TEST
f = MakeItTalk()
# f.generate_inference('./examples/cesi.jpg','./examples/M6_04_16k.wav')
f.generate_inference(opt_parser.img_path, opt_parser.audio_path)
