# -*- coding: UTF-8 -*-

"""
RNN-LSTM Recurrent Neural Network
"""

import tensorflow as tf


# Neural network model
def network_model(inputs, num_pitch, weights_file=None):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(
        512,  # Number of LSTM layer neurons (512), which is also the output dimension of the LSTM layer
        input_shape=(inputs.shape[1], inputs.shape[2]),  # Input shape, must be set for the first LSTM layer
        # return_sequences: Controls the return type
        # - True: Returns all output sequences
        # - False: Returns only the last output of the sequence
        # Must be set when stacking LSTM layers. Not necessary for the last LSTM layer.
        return_sequences=True  # Returns all output sequences
    ))
    model.add(tf.keras.layers.Dropout(0.3))  # Drop 30% of neurons to prevent overfitting
    model.add(tf.keras.layers.LSTM(512, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(512))  # return_sequences is False by default, returns only the last output
    model.add(tf.keras.layers.Dense(256))  # Fully connected layer with 256 neurons
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(num_pitch))  # Output size equals the number of unique pitches
    model.add(tf.keras.layers.Activation('softmax'))  # Softmax activation function for probability calculation
    
    # Calculate error using categorical crossentropy and RMSProp optimizer, which is good for RNNs
    # Error calculation: First use Softmax to calculate percentage probabilities, then use cross-entropy 
    # to calculate the error between the percentage probabilities and the corresponding one-hot codes
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    if weights_file is not None:  # When generating music
        # Load all neural network layer weights from the HDF5 file
        model.load_weights(weights_file)

    return model