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
    """
    A class to load BIDS-compliant EEG, audio, and event data for a specific subject and session.

    Attributes:
        subject_id (str): The subject identifier.
        session_id (str): The session identifier.
        audio_filename (str): The filename for the audio recording.
        audio_events_filename (str): The filename for the audio event markers.
        eeg_filename (str): The filename for the EEG data.
        eeg_data (mne.io.Raw): The loaded EEG data.
        audio_sr (int): The sample rate of the audio data.
        audio_data (numpy.ndarray): The loaded audio data.
        audio_events (pd.DataFrame): The loaded audio event markers.
    """
    def __init__(self, subject_id, session_id):
        """
        Initializes the BidsFileLoader with subject and session IDs and prepares file paths.

        Args:
            subject_id (str): The subject identifier.
            session_id (str): The session identifier.
        """
        printSectionHeader(f"{Fore.CYAN}📂 Reading  Subject Info (BIDS)")
        self.subject_id = subject_id
        self.session_id = session_id
        self.audio_filename = f'sub-{subject_id}_ses-{session_id}_task-PilotStudy_run-01_audio.wav'
        self.audio_events_filename = f'sub-{subject_id}_ses-{session_id}_task-PilotStudy_run-01_events.tsv'
        self.eeg_filename = f'sub-{subject_id}_ses-{session_id}_task-PilotStudy_run-01_eeg.edf'
        
        # Loading the data
        print("🔄 Initializing BIDS File Loader...")
        self.load_files()

    def load_files(self):
        """
        Loads EEG, audio, and event data files for the specified subject and session.
        
        This method locates the files within a BIDS-compliant directory structure, loads the EEG data,
        reads the audio data along with its sample rate, and parses the audio event markers.
        """
        print(f"{Fore.CYAN}📂 Loading BIDS Data...")

        # File paths
        folder = Path(config.BIDS_DIR, f'sub-{self.subject_id}', f'ses-{self.session_id}')
        eeg_filepath = Path(folder, 'eeg', self.eeg_filename)
        audio_filepath = Path(folder, 'audio', self.audio_filename)
        audio_events_filepath = Path(folder, 'audio', self.audio_events_filename)
        
        # Loading EEG data
        print(f"🧠 Loading EEG Data from {eeg_filepath}")
        self.eeg_data = mne.io.read_raw_edf(eeg_filepath, preload=True)

        # Loading Audio Data
        print(f"🎵 Loading Audio Data from {audio_filepath}")
        self.audio_sr, self.audio_data = wavfile.read(audio_filepath)

        # Loading Audio Events
        print(f"📜 Loading Audio Events from {audio_events_filepath}")
        self.audio_events = pd.read_csv(audio_events_filepath, delimiter='\t')

        print("✅ Data Loading Complete!")
