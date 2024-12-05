import os
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import config as config

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

class ModelTrainer:
    def __init__(self, model, model_name, subject_id, session_id):
        self.model = model
        self.model_name = model_name
        self.subject_id = subject_id
        self.session_id = session_id
        self.model_dir = Path(
            config.TRAINED_MODEL_DIR, 
            f'sub-{subject_id}',
            f'ses-{session_id}',
            model_name
        )
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_model_filepath = Path(self.model_dir, f'{model_name}.keras')
        
    def train(self, x_train, y_train):       
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=rmse_loss,
            metrics=['mse']
        )
        
        self.model.summary()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.best_model_filepath), 
                save_best_only=True, 
                monitor='val_loss', 
                verbose=1
            )
        ]
        
        history = self.model.fit(
            x_train, y_train,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        self.history = pd.DataFrame(history.history)
        history_filepath = Path(self.model_dir, f'{self.model_name}_history.csv')
        self.history.to_csv(history_filepath, index=False)
        self.model.save(Path(self.model_dir, f'{self.model_name}_full_model.h5'))

    def evaluate(self, x, y):
        loss, mse = self.model.evaluate(x, y)
        rmse = tf.sqrt(mse)
        print('Testing Performance')
        print(f'Loss: {loss}, RMSE: {rmse}')
