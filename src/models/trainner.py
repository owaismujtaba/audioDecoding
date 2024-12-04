import os
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import config as config

def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))



class ModelTrainner:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.model_dir = Path(config.TRAINED_MODEL_DIR, model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_model_filepath = Path(self.model_dir, model_name)
        

    def train(self, x_train, y_train):       
        self.model.compile(
            optimizer=Adam,
            loss = rmse_loss,
            metrics=['mse']
        )
        
        
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.best_model_filepath}.h5', 
                save_best_only=True, 
                monitor='val_loss', 
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            x_train, y_train,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        self.history = pd.DataFrame(self.history.history)
        histroy_filepath = Path(self.model_dir, f'{self.model_name}_history.csv')
        self.history.to_csv(histroy_filepath, index=False)

    def evaluate(self, x, y):
        loss, mse = self.model.evaluate(x, y)
        print('Testing Performance')
        print(f'Loss: {loss}, MSE: {mse}')