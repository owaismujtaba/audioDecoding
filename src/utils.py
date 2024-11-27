import numpy as np
import config as config

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




