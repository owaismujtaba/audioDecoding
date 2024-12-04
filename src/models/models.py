import tensorflow as tf
from tensorflow.keras import layers, Model

class EEGToAudioModel(tf.keras.Model):
    def __init__(self):
        super(EEGToAudioModel, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool = layers.MaxPooling2D((2, 2))
        
        # Recurrent layers for temporal modeling
        self.gru = layers.Bidirectional(layers.GRU(128, return_sequences=False))
        
        # Dense layers for output mapping
        self.dense1 = layers.Dense(512, activation='relu')
        self.dense2 = layers.Dense(9600, activation='linear')
    
    def call(self, inputs):
        # Add a channel dimension to inputs for Conv2D
        x = tf.expand_dims(inputs, -1)
        
        # Feature extraction
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        
        # Flatten and reshape for GRU input
        x = tf.reshape(x, (tf.shape(x)[0], -1, x.shape[-1]))
        
        # Temporal modeling
        x = self.gru(x)
        
        # Output mapping
        x = self.dense1(x)
        output = self.dense2(x)
        
        return output