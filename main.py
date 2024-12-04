import config as config
import pdb

if __name__=='__main__':
    

    if config.TRAIN_MODEL:
        from src.dataset.data_extractor import train_val_test_dataloader_pipeline
        from src.models.trainner import ModelTrainner
        from src.models.models import EEGToAudioModel

        train_eeg_windows, test_eeg_windows, train_audio_windows, test_audio_windows = train_val_test_dataloader_pipeline(
            subject_id='01', session_id='01'
        )
        
        model = EEGToAudioModel()
        trainner = ModelTrainner(
            model=model,
            model_name='ist'
        )
        pdb.set_trace()
        
