import speech_recognition as sr
import os
import sys
import soundfile as sf
BASE_DIR = os.environ['PWD']
GTTS_DIR = sys.path[0]
import argparse
parser = argparse.ArgumentParser(description='Inference code for glow tts')

parser.add_argument('--audio', 
                    help='text to inference on', required=True)
parser.add_argument('--outfile', type=str, 
                    help='outfile name', required=True)

args = parser.parse_args()                    
args.outfile = os.path.join(BASE_DIR, args.outfile)
AUDIO_FILE = (args.audio)
  
# use the audio file as the audio source
  
r = sr.Recognizer()
  
with sr.AudioFile(AUDIO_FILE) as source:
    #reads the audio file. Here we use record instead of
    #listen
    audio = r.record(source)  
  
try:
    content = r.recognize_google(audio)
  
except sr.UnknownValueError:
    content = "Google Speech Recognition could not understand audio"
  
except sr.RequestError as e:
    content = "Could not request results from Google Speech Recognition service; {0}".format(e)

args.outfile = os.path.join(BASE_DIR, args.outfile)
f = open(args.outfile, 'w')
f.write(content)
f.close()
