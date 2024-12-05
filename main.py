import config as config
import pdb

if __name__=='__main__':
    

    if config.TRAIN_MODEL:
        from src.dataset.data_extractor import train_val_test_dataloader_pipeline
        from src.models.trainner import ModelTrainer
        from src.models.models import EEGToAudioModel
        subject_id, session_id = '01', '01'
        train_eeg_windows, test_eeg_windows, train_audio_windows, test_audio_windows = train_val_test_dataloader_pipeline(
            subject_id=subject_id, session_id=session_id
        )
        
        model = EEGToAudioModel()
        trainner = ModelTrainer(
            model=model,
            model_name='ist',
            subject_id = subject_id,
            session_id = session_id
        )
        trainner.train(train_eeg_windows, train_audio_windows)
        
        
