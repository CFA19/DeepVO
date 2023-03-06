# -*- coding: utf-8 -*-
# Import Libraries:
from tensorflow import keras


class FlowNet(keras.Model):
    def __init__(self):
        super(FlowNet, self).__init__()
        self.flow_net = keras.models.load_model('checkpoints/flownet_h.h5')

    def call(self, inputs, **kwargs):
        x = self.flow_net(inputs)
        return x
