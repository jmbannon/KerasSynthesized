import keras
from collections import Counter

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

def keras_apps():
	return [
		'xception', 'vgg19', 'vgg16', 'resnet50', 'inception_v3', 
		'inception_resnet_v2', 'mobilenet', 'densenet121', 'densenet169', 
		'densenet201', 'nasnet_large', 'nasnet_mobile'
		]


# def evaluate_models():
	# for model in KERAS_MODELS.values():
	# 	model_layer_types = {}
	# 	for layer in model.layers:
	# 		ltype = layer.__class__.__name__
	# 		model_layer_types[ltype] = 1
	# 	for key in model_layer_types.keys():
	# 		if key in layer_types:
	# 			layer_types[key] = layer_types[key] + 1
	# 		else:
	# 			layer_types[key] = 1

	# print(layer_types)
	# layer_types_set = list(layer_types)
	# layer_types_set.sort()
	# print(layer_types_set)

	# layer_types = list(layer_types)
	# layer_types.sort()
	# print(layer_types)


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
	print('kernel_strides')
	print(Counter(kernel_strides).keys())
	print(Counter(kernel_strides).values())
	print('pooling_sizes')
	print(Counter(pooling_sizes).keys())
	print(Counter(pooling_sizes).values())
	print('pooling_strides')
	print(Counter(pooling_strides).keys())
	print(Counter(pooling_strides).values())
	print('layer_dims')
	print(Counter(layer_dims).keys())
	print(Counter(layer_dims).values())
	print('fc_count')
	print(fc_count)


	# layer_types_set = list(set(layer_types))
	# layer_types_set.sort()

	# print(layer_types_set)

	# layer_types = list(layer_types)
	# layer_types.sort()
	# print(layer_types)
	# print(list(layer_types).sort())
	# keras.utils.print_summary(keras_app(model), line_length=200)
	# keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')

# evaluate_model(keras_model('inception_v3'))
evaluate_model('vgg16')
# evaluate_model('nasnet_mobile')

