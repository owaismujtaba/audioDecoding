import tensorflow as tf
from tensorflow.keras import layers, Model


import tensorflow as tf
from tensorflow.keras import layers, Model

class EEGToAudioTransformer(tf.keras.Model):
    def __init__(self, num_channels=64, time_steps=200, num_heads=4, dff=512, num_layers=4, output_dim=9600):
        super(EEGToAudioTransformer, self).__init__()
        
        # Reshape and Linear layer for channel reduction
        self.flatten_channels = layers.Dense(time_steps, activation='linear')  # Compress channel dimension
        
        # Positional Encoding for time steps
        self.positional_encoding = PositionalEncoding(time_steps)
        
        # Transformer Encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(time_steps, num_heads, dff)
            for _ in range(num_layers)
        ]

        # Output layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.output_layer = layers.Dense(output_dim, activation='linear')

    def call(self, inputs):
        # Flatten across channels
        x = tf.reshape(inputs, [-1, tf.shape(inputs)[2], tf.shape(inputs)[1]])  # (batch_size, time_steps, channels)
        x = self.flatten_channels(x)  # Compress channels into time steps
        
        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through transformer encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # Flatten and map to output
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.output_layer(x)
        return output


class PositionalEncoding(layers.Layer):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(0, seq_len, dtype=tf.float32)[:, tf.newaxis]
        dimensions = tf.range(0, self.d_model, 2, dtype=tf.float32) / tf.cast(self.d_model, tf.float32)
        angle_rads = positions * tf.math.exp(-tf.math.log(10000.0) * dimensions)

        pos_encoding = tf.concat([tf.math.sin(angle_rads), tf.math.cos(angle_rads)], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return inputs + pos_encoding


class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


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