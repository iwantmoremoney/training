import numpy as np
from os import path
import re

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout

class AbstractModelBuilder(object):

	def __init__(self, weights_path = None):
		self.weights_path = weights_path

	def getModel(self):
		weights_path = self.weights_path
		model = self.buildModel()

		if weights_path and path.isfile(weights_path):
			try:
				model.load_weights(weights_path)
			except Exception, e:
				print e

		return model

	# You need to override this method.
	def buildModel(self):
		raise NotImplementedError("You need to implement your own model.")

class MarketModelBuilder(AbstractModelBuilder):
    def name(self):
        return path.splitext(path.basename(__file__))[0] 
    
    def buildModel(self):
        from keras.models import Model
        from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, Merge, merge
        from keras.layers.advanced_activations import LeakyReLU


        dr_rate = 0.0

        B = Input(shape = (3,))
        b = Dense(5, activation = "relu")(B)

        inputs = [B]
        merges = [b]

        for i in xrange(1):
            S = Input(shape=[2, 60, 1])
            inputs.append(S)

            h = Convolution2D(1024, 40, 1, border_mode = 'valid')(S)
            h = LeakyReLU(0.001)(h)

            h = Flatten()(h)
            h = Dense(2048)(h)
            h = LeakyReLU(0.001)(h)
            h = Dropout(dr_rate)(h)
            merges.append(h)

            h = Convolution2D(2048, 60, 1, border_mode = 'valid')(S)
            h = LeakyReLU(0.001)(h)

            h = Flatten()(h)
            h = Dense(4096)(h)
            h = LeakyReLU(0.001)(h)
            h = Dropout(dr_rate)(h)
            merges.append(h)

        m = merge(merges, mode = 'concat', concat_axis = 1)
        m = Dense(1024)(m)
        m = LeakyReLU(0.001)(m)
        m = Dropout(dr_rate)(m)
        m = Dense(512)(m)
        m = LeakyReLU(0.001)(m)
        m = Dropout(dr_rate)(m)
        m = Dense(256)(m)
        m = LeakyReLU(0.001)(m)
        m = Dropout(dr_rate)(m)
        V = Dense(2, activation = 'linear', init = 'zero')(m)
        model = Model(input = inputs, output = V)

        return model
