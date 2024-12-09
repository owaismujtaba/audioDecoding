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
WINDOW_SIZE = 0.2
FRAME_SHIFT = 0.1

TRAIN_MODEL = True
TRAINED_MODEL_DIR = Path(CUR_DIR, 'Models')
AUDIO_DIR = Path(CUR_DIR, 'Audios')

EPOCHS = 100
BATCH_SIZE = 64


