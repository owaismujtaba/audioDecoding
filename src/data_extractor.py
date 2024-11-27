import numpy as np
from colorama import Fore, Style, init

import config as config
from src.utils import filter_events, printSectionHeader
from src.bids_reader import BidsFileLoader

init()
class DataExtractor:
    """
    Extracts time-aligned EEG and audio data segments based on event annotations.
    
    This class provides functionality to:
    1. Filter events of interest from EEG annotations and audio events
    2. Extract EEG and audio data for specified time segments
     
    It is designed to work with EEG data from the MNE library and raw audio data.
    """
    def __init__(self, eeg, audio, audio_events):
        """
        Initializes the DataExtractor object with EEG, audio, and event data.
        Automatically processes the data by extracting time-aligned samples.

        Parameters:
        - eeg: An MNE object containing EEG data and annotations
        - audio: A NumPy array containing the raw audio signal
        - audio_events: A dataframe containing audio event information, including onsets (tsv file)
        """
        self.eeg = eeg
        self.audio = audio
        self.events = audio_events

        self.extract_data()


    def extract_data(self):
        """
        Processes the EEG and audio data to extract samples aligned with event onsets.
        
        Uses the `filter_events` function to identify relevant events and then extracts 
        segments of EEG and audio data corresponding to the specified time segment.
        """
        printSectionHeader(f"{Fore.CYAN}📂 Loading BIDS Subject Info")

        eeg_data = self.eeg.get_data()
        intrested_indexes = filter_events(self.eeg, self.events)
        annotations = self.eeg.annotations
        eeg_samples, auido_samples = [], []
        
        for index in intrested_indexes:
            eeg_onset = annotations[index]['onset']
            audio_onset = self.events['onset'][index]
            
            eeg_start_index = int(eeg_onset*config.EEG_SR)
            eeg_end_index = eeg_start_index + config.TIME_SEGMENT*config.EEG_SR
            eeg_samples.append(eeg_data[:,eeg_start_index:eeg_end_index])

            audio_strat_index = int(audio_onset*config.AUDIO_SR)
            audio_end_index = audio_strat_index + config.TIME_SEGMENT*config.AUDIO_SR
            auido_samples.append(self.audio[audio_strat_index: audio_end_index])


        self.eeg_samples =  np.array(eeg_samples)
        self.audio_samples = np.array(auido_samples)

class FeatureExtraction1:
    def __init__(self, eeg_samples, audio_samples):
        self.eeg_samples = eeg_samples
        self.audio_samples = audio_samples
        self.window_size = 0.2
        self.frame_shift = 0.1
        
        self.eeg_window_size = int(config.EEG_SR * self.window_size)
        self.eeg_step_size = int(config.EEG_SR * self.frame_shift)
        
        self.audio_window_size = int(config.AUDIO_SR * self.window_size)
        self.audio_step_size = int(config.AUDIO_SR * self.frame_shift)

        eeg_windows = np.array([
            self.sliding_window(trial, self.eeg_window_size, self.eeg_step_size) 
            for trial in self.eeg_samples
        ])
        audio_windows = np.array([
            self.sliding_window(trial, self.audio_window_size, self.audio_step_size) 
            for trial in self.audio_samples
        ])

        eeg_shape, audio_shape = eeg_windows.shape, audio_windows.shape
        self.eeg_windows = eeg_windows.reshape(
            eeg_shape[0]*eeg_shape[1], eeg_shape[2], eeg_shape[2]
        )
        self.audio_windows = audio_windows.reshape(
            audio_shape[0]*audio_shape[1], audio_shape[2], audio_shape[2]
        )

    def sliding_window(self, data, window_size, step_size):
        """
        Create sliding windows from data with specified window size and step size.
        """
        num_windows = (data.shape[-1] - window_size) // step_size + 1
        return np.array([
            data[..., i * step_size: i * step_size + window_size]
            for i in range(num_windows)
        ])


import numpy as np
import config

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

    def __init__(self, eeg_samples, audio_samples):
        """
        Initialize the feature extraction process.

        Args:
            eeg_samples (numpy.ndarray): EEG data with shape [N_trials, N_channels, N_timepoints].
            audio_samples (numpy.ndarray): Audio data with shape [N_trials, N_timepoints].
        """
        self.eeg_samples = eeg_samples
        self.audio_samples = audio_samples
        self.window_size = 0.2  # in seconds
        self.frame_shift = 0.1  # in seconds

        self.eeg_window_size = int(config.EEG_SR * self.window_size)
        self.eeg_step_size = int(config.EEG_SR * self.frame_shift)

        self.audio_window_size = int(config.AUDIO_SR * self.window_size)
        self.audio_step_size = int(config.AUDIO_SR * self.frame_shift)

        self.eeg_windows = self.extract_windows(self.eeg_samples, self.eeg_window_size, self.eeg_step_size)
        self.audio_windows = self.extract_windows(self.audio_samples, self.audio_window_size, self.audio_step_size)

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
        windows = np.array([
            self.sliding_window(trial, window_size, step_size)
            for trial in samples
        ])
        # Combine trials and windows into a single axis
        combined_shape = (-1, *windows.shape[2:])  # Flatten first two dimensions
        return windows.reshape(combined_shape)


def data_extraction_pipeline(subject_id, session_id):
    bids_reader = BidsFileLoader(subject_id=subject_id, session_id=session_id)
    eeg = bids_reader.eeg_data
    audio = bids_reader.audio_data
    audio_events = bids_reader.audio_events

    data_extractor = DataExtractor(
        eeg=eeg, audio=audio, audio_events=audio_events
    )

    eeg_samples = data_extractor.eeg_samples
    audio_samples = data_extractor.audio_samples

    return eeg_samples, audio_samples