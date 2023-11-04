#pip install googletrans==3.1.0a0
#pip install -U easynmt
import os
import sys
from googletrans import Translator
#from easynmt import EasyNMT
#model = EasyNMT('opus-mt')
translator = Translator(service_urls=['translate.googleapis.com'])
languages={'bn': 'bengali','hi': 'hindi','ml': 'malayalam','mr': 'marathi','ta': 'tamil','te': 'telugu'}
print(os.environ)
BASE_DIR = os.environ['PWD']
GTTS_DIR = sys.path[0]
import argparse
parser = argparse.ArgumentParser(description='Inference code for glow tts')

parser.add_argument('--text', type=str, 
                    help='text to inference on', required=True)
parser.add_argument('--lang', type=str, 
                    help='text to inference on', required=True)                    
parser.add_argument('--outfile', type=str, 
                    help='outfile name', required=True)

args = parser.parse_args()                    
args.outfile = os.path.join(BASE_DIR, args.outfile)
f =open(args.text)
translated = translator.translate(f.read(), dest=args.lang)
print(f.read())
print(translated.text,"traslated")
#x = model.translate(f.read(), target_lang=args.lang)
#print(x,"hello")
f.close()
f = open(args.outfile, 'w')
f.write(translated.text)
f.close()
