from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

import numpy as np
from collections import Counter

import types

def to_cpp_tensor(readable_tensor, strip_channels=True):
	tensor = readable_tensor
	if isinstance(readable_tensor, list):
		tensor = readable_tensor[0]

	if strip_channels and len(tensor.shape) == 4 and tensor.shape[0] == 1:
		tensor = tensor[0]

	declaration = 'Numeric tensor' + str(tensor.shape)
	declaration = declaration.replace('(', '[')
	declaration = declaration.replace(', ', '][')
	declaration = declaration.replace(')', ']')

	tstr = str(repr(tensor))
	tstr = tstr.replace(')', '')
	tstr = tstr.replace('array(', '')
	tstr = tstr.replace('[', '{')
	tstr = tstr.replace(']', '}')

	return '{} =\n      {};'.format(declaration, tstr)

def list_lambda(func, value):
	if isinstance(value, list):
		return [func(x) for x in value]
	else:
		return func(value)

# Translates from Keras Default
# (rows, cols, depth, nr_filters)
#   to
# (nr_filters, depth, rows, cols)
def to_readable_weight(tensor):
	def to_readable_arr(arr):
		arr = np.swapaxes(arr, 3, 0)
		arr = np.swapaxes(arr, 2, 1)
		arr = np.swapaxes(arr, 2, 3)
		return arr
	return list_lambda(to_readable_arr, tensor)

# Translates from readable
# (nr_filters, depth, rows, cols)
#   to
# (rows, cols, depth, nr_filters)
def to_keras_weight(tensor):
	def to_keras_arr(arr):
		arr = np.swapaxes(arr, 0, 3)
		arr = np.swapaxes(arr, 1, 2)
		arr = np.swapaxes(arr, 0, 1)
		return arr
	return list_lambda(to_keras_arr, tensor)


###########################################


# Translates from readable
# (nr_inputs, depth, rows, cols)
#   to
# (nr_inputs, rows, cols, depth)
def to_keras_tensor(tensor):
	def to_keras_arr(arr):
		arr = np.swapaxes(arr, 1, 2)
		arr = np.swapaxes(arr, 3, 2)
		return arr
	return list_lambda(to_keras_arr, tensor)

# Translates from Keras
# (nr_inputs, rows, cols, depth)
#   to
# (nr_inputs, depth, rows, cols)
def to_readable_tensor(tensor):
	def to_readable_arr(arr):
		arr = np.swapaxes(arr, 3, 1)
		arr = np.swapaxes(arr, 2, 3)
		return arr
	return list_lambda(to_readable_arr, tensor)

readable_input = np.array(
	[[
	  [
	    [ 0., 1., 2., 3., 4. ],
        [ 5., 6., 7., 8., 9. ],
        [ 10., 11., 12., 13., 14. ],
        [ 15., 16., 17., 18., 19. ],
        [ 20., 21., 22., 23., 24. ]
      ],
	  [
	    [ 0., 1., 2., 3., 4. ],
        [ 5., 6., 7., 8., 9. ],
        [ 10., 11., 12., 13., 14. ],
        [ 15., 16., 17., 18., 19. ],
        [ 20., 21., 22., 23., 24. ]
      ],
	  [
	    [ 0., 1., 2., 3., 4. ],
        [ 5., 6., 7., 8., 9. ],
        [ 10., 11., 12., 13., 14. ],
        [ 15., 16., 17., 18., 19. ],
        [ 20., 21., 22., 23., 24. ]
      ],
    ]]
  )


test_weights = [np.array(
	[
	  [
	    [ [0., 0.], [0., 0.], [0., 0.] ],
        [ [1., 1.], [1., 1.], [1., 1.] ],
        [ [2., 2.], [2., 2.], [2., 2.] ],
      ],
	  [
	    [ [3., 3.], [3., 3.], [3., 3.] ],
        [ [4., 4.], [4., 4.], [4., 4.] ],
        [ [5., 5.], [5., 5.], [5., 5.] ],
      ],
      [
	    [ [6., 6.], [6., 6.], [6., 6.] ],
        [ [7., 7.], [7., 7.], [7., 7.] ],
        [ [8., 8.], [8., 8.], [8., 8.] ],
      ]
    ]
  )]

readable_test_weights = [np.array(
	[
	  [ # Kernel 1
		[
		    [ 0., 1., 2. ],
	        [ 3., 4., 5. ],
	        [ 6., 7., 8. ]
      	],
      	[
		    [ 0., 1., 2. ],
	        [ 3., 4., 5. ],
	        [ 6., 7., 8. ]
      	],
      	[
		    [ 0., 1., 2. ],
	        [ 3., 4., 5. ],
	        [ 6., 7., 8. ]
      	],
      ],
      [ # Kernel 2
		[
		    [ 0., 1., 2. ],
	        [ 3., 4., 5. ],
	        [ 6., 7., 8. ]
      	],
      	[
		    [ 0., 1., 2. ],
	        [ 3., 4., 5. ],
	        [ 6., 7., 8. ]
      	],
      	[
		    [ 0., 1., 2. ],
	        [ 3., 4., 5. ],
	        [ 6., 7., 8. ]
      	],
      ]

	]

)]

# Channels last is DEFAULT
# Input shape: (rows, cols, depth)
# 
# Weights: [(rows, cols, depth, nr_filters)]

model = Sequential()
# weights = [np.ones((3, 3, 3, 1))]
weights = test_weights
# weights = to_keras_tensor(readable_test_weights)

print(to_cpp_tensor(readable_test_weights))
print(to_cpp_tensor(readable_input))
# print(to_keras_weight(readable_test_weights))
test_layer = Conv2D(2, (3, 3), input_shape=(5, 5, 3), weights=weights, use_bias=False, name='conv')
# test_layer.set_weights(test_weights_2)

# print(test_layer.get_weights())

# test_layer.set_weights(test_weights)
# print(test_layer)

model.add(test_layer)
model.compile(optimizer='sgd', loss='mean_squared_error')

out = model.predict(to_keras_tensor(readable_input))

print(to_readable_tensor(out))




