import numpy as np
from sklearn.model_selection import train_test_split
from colorama import Fore, Style, init

import config as config
from src.utils import filter_events, printSectionHeader
from src.utils import normalize_audio, normalize_eeg
from src.dataset.bids_reader import BidsFileLoader

import pdb

init()

class DataExtractor:
    """
    Extracts time-aligned EEG and audio data segments based on event annotations.
    
    This class provides functionality to:
    1. Filter events of interest from EEG annotations and audio events
    2. Extract EEG and audio data for specified time segments
     
    It is designed to work with EEG data from the MNE library and raw audio data.
    """
    def __init__(self, eeg, audio, audio_events, subject_id, session_id):
        """
        Initializes the DataExtractor object with EEG, audio, and event data.
        Automatically processes the data by extracting time-aligned samples.

        Parameters:
        - eeg: An MNE object containing EEG data and annotations
        - audio: A NumPy array containing the raw audio signal
        - audio_events: A dataframe containing audio event information, including onsets (tsv file)
        """
        printSectionHeader(f"🔄 Initializing Data Extractor for sub-{subject_id} ses-{session_id}")

        self.eeg = eeg
        self.audio = audio
        self.events = audio_events
        self.subject_id = f'sub-{subject_id}'
        self.session_id = f'ses-{session_id}'

        self.extract_data()

    def extract_data(self):
        """
        Processes the EEG and audio data to extract samples aligned with event onsets.
        
        Uses the `filter_events` function to identify relevant events and then extracts 
        segments of EEG and audio data corresponding to the specified time segment as 
        specified in config.py.
        """
        print(f"📂 Loading EEG and Audio Data ")

        eeg_data = self.eeg.get_data()  # 🧠 EEG Data
        intrested_indexes = filter_events(self.eeg, self.events)  # 🎯 Filtering Events
        annotations = self.eeg.annotations  # 🧾 EEG Annotations
        eeg_samples, audio_samples = [], []

        # Extract the samples based on event onsets
        print("🔍 Extracting EEG and Audio Segments...")

        for index in intrested_indexes:
            eeg_onset = annotations[index]['onset']  # 🧠 EEG Onset
            audio_onset = self.events['onset'][index]  # 🎵 Audio Onset
            
            # Get EEG samples for the specified time window
            eeg_start_index = int(eeg_onset * config.EEG_SR)
            eeg_end_index = eeg_start_index + config.TIME_SEGMENT * config.EEG_SR
            eeg_samples.append(eeg_data[:, eeg_start_index:eeg_end_index])

            # Get Audio samples for the specified time window
            audio_start_index = int(audio_onset * config.AUDIO_SR)
            audio_end_index = audio_start_index + config.TIME_SEGMENT * config.AUDIO_SR
            audio_samples.append(self.audio[audio_start_index: audio_end_index])

        # Store the extracted samples
        self.eeg_samples = np.array(eeg_samples)  # 🧠 Extracted EEG Samples
        self.eeg_samples = normalize_eeg(self.eeg_samples)
        self.audio_samples = np.array(audio_samples)  # 🎵 Extracted Audio Samples
        self.audio_samples = normalize_audio(self.audio_samples, self.subject_id, self.session_id)
        print("✅ Data Extraction Complete!")

class FeatureExtraction:
    """
    Extract sliding windows from EEG and audio samples for feature analysis.

    Attributes:
        eeg_samples (numpy.ndarray): The EEG data, expected shape [N_trials, N_channels, N_timepoints].
        audio_samples (numpy.ndarray): The audio data, expected shape [N_trials, N_timepoints].
        window_size (float): Window size in seconds.
        frame_shift (float): Frame shift (step size) in seconds.
        eeg_windows (numpy.ndarray): Extracted EEG sliding windows.
        audio_windows (numpy.ndarray): Extracted audio sliding windows.
    """

    def __init__(self, eeg_samples, audio_samples, subject_id, session_id):
        """
        Initialize the feature extraction process.

        Args:
            eeg_samples (numpy.ndarray): EEG data with shape [N_trials, N_channels, N_timepoints].
            audio_samples (numpy.ndarray): Audio data with shape [N_trials, N_timepoints].
        """
        printSectionHeader(f"🔄 Initializing Feature Extraction for sub-{subject_id} ses-{session_id} ")

        self.eeg_samples = eeg_samples
        self.audio_samples = audio_samples
        self.window_size = config.WINDOW_SIZE  # in seconds
        self.frame_shift = config.FRAME_SHIFT  # in seconds

        self.eeg_window_size = int(config.EEG_SR * self.window_size)
        self.eeg_step_size = int(config.EEG_SR * self.frame_shift)

        self.audio_window_size = int(config.AUDIO_SR * self.window_size)
        self.audio_step_size = int(config.AUDIO_SR * self.frame_shift)

        print(f"🔧 EEG Window Size: {self.eeg_window_size} samples, Step Size: {self.eeg_step_size} samples")
        print(f"🔧 Audio Window Size: {self.audio_window_size} samples, Step Size: {self.audio_step_size} samples")

        # Extract windows
        print("🔍 Extracting EEG and Audio sliding windows...")
        self.eeg_windows = self.extract_windows(self.eeg_samples, self.eeg_window_size, self.eeg_step_size)
        self.audio_windows = self.extract_windows(self.audio_samples, self.audio_window_size, self.audio_step_size)

        printSectionHeader("✅ Feature Extraction Complete!")

    def sliding_window(self, data, window_size, step_size):
        """
        Create sliding windows from 1D or multidimensional data.

        Args:
            data (numpy.ndarray): The input data. Last dimension should be time.
            window_size (int): The size of each window (in samples).
            step_size (int): The step size (in samples).

        Returns:
            numpy.ndarray: Extracted sliding windows with shape [..., num_windows, window_size].
        """
        num_windows = (data.shape[-1] - window_size) // step_size + 1
        if num_windows < 1:
            raise ValueError("Window size and step size are too large for the data.")
        return np.array([
            data[..., i * step_size: i * step_size + window_size]
            for i in range(num_windows)
        ])

    def extract_windows(self, samples, window_size, step_size):
        """
        Extract sliding windows for all trials.

        Args:
            samples (numpy.ndarray): Input data with trials as the first dimension.
            window_size (int): Size of each window (in samples).
            step_size (int): Step size between windows (in samples).

        Returns:
            numpy.ndarray: Reshaped array of sliding windows.
        """
        print("🔄 Extracting windows for all trials...")
        windows = np.array([
            self.sliding_window(trial, window_size, step_size)
            for trial in samples
        ])
        combined_shape = (-1, *windows.shape[2:])  # Flatten first two dimensions
        print("✅ Windows Extraction Complete!")
        return windows.reshape(combined_shape)

def data_extraction_pipeline(subject_id, session_id):
    printSectionHeader("🔄 Starting Data Extraction Pipeline...")

    # 📂 Load BIDS Data
    print(f"📂 Loading EEG and Audio data for sub-{subject_id}, ses-{session_id}...")
    bids_reader = BidsFileLoader(subject_id=subject_id, session_id=session_id)
    eeg = bids_reader.eeg_data  # 🧠 EEG Data
    audio = bids_reader.audio_data  # 🎵 Audio Data
    audio_events = bids_reader.audio_events  # 📍 Audio Events
   
    printSectionHeader("✅ Data Read Successfully!")

    # 🔍 Extract Data
    print("🔍 Extracting EEG and Audio samples...")
    data_extractor = DataExtractor(
        eeg=eeg, audio=audio, audio_events=audio_events,
        subject_id=subject_id, session_id=session_id
    )
    eeg_samples = data_extractor.eeg_samples  # 🧠 Extracted EEG Samples
    audio_samples = data_extractor.audio_samples  # 🎵 Extracted Audio Samples
    
    printSectionHeader("✅ Data Extraction Complete!")

    return eeg_samples, audio_samples


def train_val_test_dataloader_pipeline(subject_id, session_id):
    printSectionHeader("🔄 Starting Train-Val-Test Data Pipeline...")

    # Get EEG and Audio Samples
    print(f"🔄 Extracting data for Subject {subject_id}, Session {session_id}...")
    eeg_samples, audio_samples = data_extraction_pipeline(
        subject_id=subject_id, session_id=session_id
    )

    # Split Data for Training and Testing
    print("🔀 Splitting data into Train and Test sets...")
    indexs = [i for i in range(eeg_samples.shape[0])]
    train_indexs, test_indexs = train_test_split(indexs, test_size=0.15, random_state=42)

    train_eeg_samples = eeg_samples[train_indexs]
    test_eeg_samples = eeg_samples[test_indexs]

    train_audio_samples = audio_samples[train_indexs]
    test_audio_samples = audio_samples[test_indexs]

    print("✅ Data Split Complete!")

    # Feature Extraction
    print("🔧 Extracting Features from Train and Test data...")
    train_feature_extractor = FeatureExtraction(
        eeg_samples=train_eeg_samples,
        audio_samples=train_audio_samples,
        subject_id=subject_id, session_id=session_id
    )

    test_feature_extractor = FeatureExtraction(
        eeg_samples=test_eeg_samples,
        audio_samples=test_audio_samples,
        subject_id=subject_id, session_id=session_id
    )

    train_eeg_windows = train_feature_extractor.eeg_windows  # 🧠 EEG Windows
    train_audio_windows = train_feature_extractor.audio_windows  # 🎵 Audio Windows
    test_eeg_windows = test_feature_extractor.eeg_windows  # 🧠 EEG Windows
    test_audio_windows = test_feature_extractor.audio_windows  # 🎵 Audio Windows

    printSectionHeader("✅ Feature Extraction Complete for Train and Test!")

    return train_eeg_windows, test_eeg_windows, train_audio_windows, test_audio_windows

