import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import cv2
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

class mini_project_2:

	# step 1: load data
	def load_data(self):



		# DATADIR = "ItemImages"
		# CATEGORIES = ["radio", "watch"]

		DATADIR = "PetImages"
		CATEGORIES = ["Dog", "Cat"]
		self.category = CATEGORIES

		for category in CATEGORIES:  # do dogs and cats
			path = os.path.join(DATADIR,category)  # create path to dogs and cats
			for img in os.listdir(path):  # iterate over each image per dogs and cats
				img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
				# plt.imshow(img_array, cmap='gray')  # graph it
				# plt.show()  # display!

				# break  # we just want one for now so break
			# beak  #...and one more!

			# print(img_array)
			# print(img_array.shape)

		IMG_SIZE = 70

		new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
		# plt.imshow(new_array, cmap='gray')
		# plt.show()

		training_data = []
	# def create_training_data(self):
		for category in CATEGORIES:  # do dogs and cats
			path = os.path.join(DATADIR,category)  # create path to dogs and cats
			class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

			for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
				try:
					img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
					new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
					training_data.append([new_array, class_num])  # add this to our training_data
				except Exception as e:  # in the interest in keeping the output clean...
					pass

		# create_training_data()
		# print(len(training_data))


		random.shuffle(training_data)

		X = []
		y = []

		for features,label in training_data:
			X.append(features)
			y.append(label)


		X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)



		pickle_out = open("X.pickle","wb")
		pickle.dump(X, pickle_out)
		pickle_out.close()

		pickle_out = open("y.pickle","wb")
		pickle.dump(y, pickle_out)
		pickle_out.close()

	# step 2: machine learning 
	def learn_data(self):

		pickle_in = open("X.pickle","rb")
		X = pickle.load(pickle_in)

		pickle_in = open("y.pickle","rb")
		y = pickle.load(pickle_in)


		X = X/255.0

		dense_layers = [0]
		layer_sizes = [64]
		conv_layers = [3]

		for dense_layer in dense_layers:
			for layer_size in layer_sizes:
				for conv_layer in conv_layers:
					NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
					print(NAME)

					model = Sequential()

					model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
					model.add(Activation('relu'))
					model.add(MaxPooling2D(pool_size=(2, 2)))

					for l in range(conv_layer-1):
						model.add(Conv2D(layer_size, (3, 3)))
						model.add(Activation('relu'))
						model.add(MaxPooling2D(pool_size=(2, 2)))

					model.add(Flatten())

					for _ in range(dense_layer):
						model.add(Dense(layer_size))
						model.add(Activation('relu'))

					model.add(Dense(1))
					model.add(Activation('sigmoid'))

					tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

					model.compile(loss='binary_crossentropy',
								  optimizer='adam',
								  metrics=['accuracy'],
								  )

					model.fit(X, y,
							  batch_size=32,
							  epochs=1,
							  validation_split=0.3,
							  callbacks=[tensorboard])

		model.save('64x3-CNN.model')


	# step 3: Detect Data by using trained dataset	
	def test_data(self,image):

		print(image)

		IMG_SIZE = 70  # 50 in txt-based
		img_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
		new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
		x = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

		model = tf.keras.models.load_model("64x3-CNN.model")

		prediction = model.predict([x])
		print(prediction)  # will be a list in a list.
		print(self.category[int(prediction[0][0])])

if __name__ == '__main__':

	data = mini_project_2()
	image_name = raw_input('Enter the data name:')
	image = image_name + '.jpg'
	# print(image)

	data.load_data()
	data.learn_data()
	# print(image)
	data.test_data(image)


