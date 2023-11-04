import os
import sys
import soundfile as sf
from gtts import gTTS
#print(os.environ)
BASE_DIR = os.environ['PWD']
GTTS_DIR = sys.path[0]
import argparse
parser = argparse.ArgumentParser(description='Inference code for glow tts')

parser.add_argument('--text', type=str, 
                    help='text to inference on', required=True)
parser.add_argument('--outfile', type=str, 
                    help='outfile name', required=True)

args = parser.parse_args()                    
args.outfile = os.path.join(BASE_DIR, args.outfile)
tts = gTTS(args.text)
tts.save(args.outfile)
