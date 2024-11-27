import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy.io import wavfile
from colorama import Fore, Style, init

import config as config
from src.utils import printSectionHeader
import pdb
init()

class BidsFileLoader:
    def __init__(self, subject_id, session_id):
        printSectionHeader(f"{Fore.CYAN}📂 Loading BIDS Subject Info")
        self.subject_id, self.session_id = subject_id, session_id
        self.audio_filename = f'sub-{subject_id}_ses-{session_id}_task-PilotStudy_run-01_audio.wav'
        self.audio_events_filename = f'sub-{subject_id}_ses-{session_id}_task-PilotStudy_run-01_events.tsv'
        self.eeg_filename = f'sub-{subject_id}_ses-{session_id}_task-PilotStudy_run-01_eeg.edf'
        self.load_files()



    def load_files(self):
        printSectionHeader(f"{Fore.CYAN}📂 Loading BIDS Data")
        folder = Path(config.BIDS_DIR, f'sub-{self.subject_id}', f'ses-{self.session_id}')
        eeg_filepath = Path(folder, 'eeg', self.eeg_filename)
        audio_filepath = Path(folder, 'audio', self.audio_filename)
        audio_events_filepath = Path(folder, 'audio', self.audio_events_filename)
        self.eeg_data = mne.io.read_raw_edf(eeg_filepath, preload=True)
        self.audio_sr, self.audio_data = wavfile.read(audio_filepath)
        self.audio_events = pd.read_csv(audio_events_filepath, delimiter='\t')
