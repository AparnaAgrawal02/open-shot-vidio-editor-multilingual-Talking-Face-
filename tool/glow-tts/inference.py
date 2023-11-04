import sys
import os
print(os.environ)
BASE_DIR = os.environ['PWD']
GLOW_DIR = sys.path[0]
sys.path.append(os.path.join(GLOW_DIR, 'waveglow'))

import librosa
import numpy as np
import glob
import json

import torch
from text import text_to_sequence, cmudict
from text.symbols import symbols
import commons
import attentions
import modules
import models
import utils
import soundfile as sf
import tqdm

# load WaveGlow
waveglow_path = os.path.join(GLOW_DIR, 'waveglow/waveglow_256channels_ljs_v3.pt') # or change to the latest version of the pretrained WaveGlow.
# waveglow_path = '/home/anchit.gupta/FLASK/glow-tts/waveglow/waveglow_256channels_ljs_v3.pt' # or change to the latest version of the pretrained WaveGlow.
waveglow = torch.load(waveglow_path)['model']
# waveglow = waveglow.remove_weightnorm(waveglow)
_ = waveglow.cuda().eval()
from apex import amp
waveglow, _ = amp.initialize(waveglow, [], opt_level="O3") # Try if you want to boost up synthesis speed.

import argparse
parser = argparse.ArgumentParser(description='Inference code for glow tts')

parser.add_argument('--text', type=str, 
                    help='text to inference on', required=True)
parser.add_argument('--outfile', type=str, 
                    help='outfile name', required=True)
parser.add_argument('--ckpt', type=str, help='ckpt', default="G_1000.pth")
args = parser.parse_args()

args.outfile = os.path.join(BASE_DIR, args.outfile)
hps = utils.get_hparams_from_file(os.path.join(GLOW_DIR, "configs/psc.json"))
# checkpoint_path = "/home/anchit.gupta/FLASK/glow-tts/G_1000.pth"
checkpoint_path = os.path.join(GLOW_DIR, args.ckpt)

model = models.FlowGenerator(
    len(symbols) + getattr(hps.data, "add_blank", False),
    out_channels=hps.data.n_mel_channels,
    **hps.model).to("cuda")

utils.load_checkpoint(checkpoint_path, model)
model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
_ = model.eval()

cmu_dict = cmudict.CMUDict(os.path.join(GLOW_DIR, hps.data.cmudict_path))

# normalizing & type casting
def normalize_audio(x, max_wav_value=hps.data.max_wav_value):
    return np.clip((x / np.abs(x).max()) * max_wav_value, -32768, 32767).astype("int16")



tst_stn = args.text
tst_stn = tst_stn.replace(".", " . ")
tst_stn = tst_stn.replace(",", " , ")
tst_stn = tst_stn.replace("?", " ? ")
tst_stn = tst_stn.replace("!", " ! ")

if getattr(hps.data, "add_blank", False):
    text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)
    text_norm = commons.intersperse(text_norm, len(symbols))
else: 
    tst_stn = " " + tst_stn.strip() + " "
    text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)
sequence = np.array(text_norm)[None, :]

x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()

with torch.no_grad():
  noise_scale = 0.667
  length_scale = 1.0
  (y_gen_tst, *_), *_, (attn_gen, *_) = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)
  try:
    audio = waveglow.infer(y_gen_tst.half(), sigma=.666)
  except:
    audio = waveglow.infer(y_gen_tst, sigma=.666)


aud = normalize_audio(audio[0].clamp(-1,1).data.cpu().float().numpy())

sf.write(args.outfile, aud, 22050)
