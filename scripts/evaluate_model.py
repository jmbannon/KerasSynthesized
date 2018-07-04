import keras
import argparse
import re
import numpy as np
from collections import Counter

KERAS_APPS = [
	'xception', 'vgg19', 'vgg16', 'resnet50', 'inception_v3', 
	'inception_resnet_v2', 'mobilenet', 'densenet121', 'densenet169', 
	'densenet201', 'nasnet_large', 'nasnet_mobile'
]

def keras_app(name):
	if name == 'xception': 
		return keras.applications.xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	elif name == 'vgg19': 
		return keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	elif name == 'vgg16':
		return keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	elif name == 'resnet50': 
		return keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	elif name == 'inception_v3': 
		return keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=1000)
	elif name == 'inception_resnet_v2': 
		return keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=1000)
	elif name == 'mobilenet': 
		return keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
	elif name == 'densenet121': 
		return keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	elif name == 'densenet169': 
		return keras.applications.densenet.DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	elif name == 'densenet201': 
		return keras.applications.densenet.DenseNet201(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	elif name == 'nasnet_large': 
		return keras.applications.nasnet.NASNetLarge(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
	elif name == 'nasnet_mobile': 
		return keras.applications.nasnet.NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
	return None


def evaluate_model(model_name):
	model = keras_app(model_name)
	kernel_sizes = []
	kernel_strides = []
	pooling_sizes = []
	pooling_strides = []
	layer_dims = []
	fc_count = 0

	for layer in model.layers:
		layer_type = layer.__class__.__name__
		if layer_type == 'Conv2D':
			kernel_sizes.append(layer.kernel_size)
			kernel_strides.append(layer.strides)
		elif layer_type == 'MaxPooling2D' or layer_type == 'AveragePooling2D':
			pooling_sizes.append(layer.pool_size)
			pooling_strides.append(layer.strides)
		elif layer_type == 'Dense':
			fc_count += 1
		layer_dims.append(layer.output_shape)

	print('kernel_sizes')
	print(Counter(kernel_sizes).keys())
	print(Counter(kernel_sizes).values())
	print('\nkernel_strides')
	print(Counter(kernel_strides).keys())
	print(Counter(kernel_strides).values())
	print('\npooling_sizes')
	print(Counter(pooling_sizes).keys())
	print(Counter(pooling_sizes).values())
	print('pooling_strides')
	print(Counter(pooling_strides).keys())
	print(Counter(pooling_strides).values())
	print('\nlayer_dims')
	print(Counter(layer_dims).keys())
	print(Counter(layer_dims).values())
	print('\nfc_count')
	print(fc_count)
	print('\nModel Summary')
	# keras.utils.print_summary(keras_app(model), line_length=200)
	print(model.summary())

def compute_padding(conv_layer):
	np_input_2d = np.array([conv_layer.input_shape[1], conv_layer.input_shape[2]])
	np_output_2d = np.array([conv_layer.output_shape[1], conv_layer.output_shape[2]])
	np_kernel = np.array(conv_layer.kernel_size)
	np_stride = np.array(conv_layer.strides)
	np_output_2d_no_padding = (((np_input_2d - np_kernel) / np_stride) + 1)
	padding = np_output_2d - np_output_2d_no_padding
	return tuple(padding.astype(int))

def evaluate_model2(model_name):
	model = keras_app(model_name)

	configurations = []

	def format_conf(conf):
		return re.sub('\s+',' ',conf).strip()

	for layer in model.layers:
		layer_type = layer.__class__.__name__

		print(f'\n{layer_type}')
		print(f'Input Shape: {layer.input_shape}')
		if layer_type == 'Conv2D':
			padding = compute_padding(layer)
			configurations.append(format_conf(f"""
				make test-fpga
				  LAYER={layer_type} 
				  INPUT_ROWS={layer.input_shape[1]} 
				  INPUT_COLS={layer.input_shape[2]}
				  PADDING_ROWS={padding[0]}
				  PADDING_COLS={padding[1]} && 
				./test-fpga
			"""))

			print(f'Kernel Size: {layer.kernel_size}')
			print(f'Kernel Strides: {layer.strides}')
			print(f'Padding Size: {padding}')
			print(f'Filters: {layer.filters}')
		elif layer_type == 'ZeroPadding2D':
			printf(f'Padding Size: {layer.padding}')
		elif layer_type == 'MaxPooling2D' or layer_type == 'AveragePooling2D':
			configurations.append(format_conf(f"""
				make test-fpga
				  LAYER={layer_type} 
				  INPUT_ROWS={layer.input_shape[1]} 
				  INPUT_COLS={layer.input_shape[2]}
				  POOL_SIZE_ROWS={layer.pool_size[0]}
				  POOL_SIZE_COLS={layer.pool_size[1]}
				  POOL_STRIDE_ROWS={layer.strides[0]}
				  POOL_STRIDE_COLS={layer.strides[1]} && 
				./test-fpga
			"""))

			print(f'Pool Size: {layer.pool_size}')
			print(f'Pool Strides: {layer.strides}')
		elif layer_type == 'Dense':
			print(f'TODO')
		else:
			print(f'NOT SUPPORTED')

	print('\n'.join(list(set(configurations))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prints summary of Keras model')
    parser.add_argument('name', type=str, help='Name of keras model', choices=KERAS_APPS)
    args = parser.parse_args()
    evaluate_model2(args.name)

