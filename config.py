import os
from pathlib import Path

CUR_DIR = os.getcwd()
BIDS_DIR = Path(CUR_DIR, 'Data')


SPEECH_TYPE ='Real'
LANGUAGE_ELEMENT = 'Word'
EVENT_TYPE ='Experiment'
START_END = 'Start'
TRIAL_PHASE = 'Speech'
PRESENTATION_MODE=None

EEG_SR = 1000
AUDIO_SR = 48000

TIME_SEGMENT = 2