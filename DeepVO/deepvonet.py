# -*- coding: utf-8 -*-
# Import Libraries:
import tensorflow as tf
from tensorflow import keras
from keras import layers


class FlowNet(keras.Model):
    def __init__(self):
        super(FlowNet, self).__init__()
        self.flownet = tf.keras.models.load_model('/media/cfa/CFA4TB/VIT/PhD/Research/PhDTest/2023/Eduardo/DeepVO/TensorFlowImplementation/checkpoints/flownet_h.h5')
        self.flownet.trainable = False
        self.reshape = keras.layers.Reshape((-1, 20 * 6 * 1024))

    def call(self, inputs, **kwargs):
        x = self.flownet(inputs)
        x = self.reshape(x)
        return x


class DeepVONet(keras.Model):
    def __init__(self):
        super(DeepVONet, self).__init__()
        rnn_cells = [tf.keras.layers.LSTMCell(1000) for _ in range(2)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
        self.lstm_layer = tf.keras.layers.RNN(stacked_lstm)
        self.out = layers.Dense(6)

    def call(self, inputs, **kwargs):
        x = self.lstm_layer(inputs)
        x = self.out(x)
        return x




