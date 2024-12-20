import numpy as np
import os
import json
import config as config
from pathlib import Path
from scipy.io.wavfile import write
import tensorflow as tf

import pdb

def printSectionHeader(message):
    """
    Print a formatted section header.

    Args:
        message (str): The message to be displayed in the header.
    """
    print("\n" + "=" * 60)
    print(message.center(60))
    print("=" * 60 + "\n")

def check_property(property, description):
    if property == None:
        return True
    elif property in description:
        return True
    else:
        return False

def filter_events(eeg, auio_events):
    annotations = eeg.annotations
    intrested_indexes = []
    for index in range(auio_events.shape[0]):
        description = annotations[index]['description']

        if all(check_property(parm, description) for parm in [
            config.SPEECH_TYPE, config.LANGUAGE_ELEMENT, config.EVENT_TYPE,
            config.START_END, config.TRIAL_PHASE, config.PRESENTATION_MODE
        ]):
            intrested_indexes.append(index)

    return intrested_indexes


def normalize_audio(audio_data: np.ndarray, subject_id, session_id) -> np.ndarray:
    """
    Normalize np.int16 audio data to the range [0, 1].
    
    Parameters:
        audio_data (np.ndarray): Input audio data as np.int16.
        
    Returns:
        np.ndarray: Normalized audio data in the range [0, 1].
    """
    print('Normalizing Audio')
    min_val = tf.reduce_min(audio_data)
    max_val = tf.reduce_max(audio_data)
    normalized_audio = (audio_data - min_val) / (max_val - min_val)  # Scale to [0, 1]

    #max_val = np.max(np.abs(audio_data))  
    #normalized_audio = audio_data.astype(np.float32) / max_val
    
    filepath = Path(
        config.CUR_DIR, 'configuration', 
        f'{subject_id}_{session_id}_audio_info.json'
    )
    data = {
        'subject_id':subject_id,
        'session_id':session_id,
        'max_val':float(max_val),
        'min_val':float(min_val)
    }
    with open(filepath, 'w') as json_file:
        json.dump(data, json_file)

    return normalized_audio

def denormalize_audio(normalized_audio: np.ndarray,  subject_id, session_id) -> np.ndarray:
    """
    Convert normalized audio data back to np.int16 scale.
    
    Parameters:
        normalized_audio (np.ndarray): Normalized audio data in the range [0, 1].
        
    Returns:
        np.ndarray: Audio data restored to np.int16 scale.
    """
    filepath = Path(
        config.CUR_DIR, 'configuration', 
        f'{subject_id}_{session_id}_audio_info.json'
    )
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    
    max_val = data["max_val"]
    min_val = data['min_val']

    normalized_audio = np.array(data["normalized_audio"], dtype=np.float32)
    restored_audio = (normalized_audio * (max_val - min_val) + min_val).astype(np.int16)
    return restored_audio


def normalize_eeg(eeg_data):
    """
    Normalize EEG data using z-score normalization.
    
    Parameters:
        eeg_data (np.ndarray): EEG data array of shape (n_samples, n_channels, n_timepoints).
        
    Returns:
        np.ndarray: Normalized EEG data of the same shape.
    """
    print('Normalizing EEG Data')
    eeg_data = np.asarray(eeg_data)
    
    mean = eeg_data.mean(axis=-1, keepdims=True)  # Mean across the last axis (timepoints)
    std = eeg_data.std(axis=-1, keepdims=True)    # Std deviation across the last axis (timepoints)
    
    std[std == 0] = 1
    normalized_data = (eeg_data - mean) / std
    
    return normalized_data

def make_audio(audio, subject_id, session_id):
    print('Wrting Audio Info File')
    dir = Path(config.AUDIO_DIR)
    filename = Path(
        dir, 
        f'{subject_id}_{session_id}.wav'
    )
    os.makedirs(dir, exist_ok=True)
    audio = audio.astype(np.int16)
    audio = audio.flatten()
   
    write(filename, config.AUDIO_SR, audio)