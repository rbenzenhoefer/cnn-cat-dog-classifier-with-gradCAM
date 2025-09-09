''' Imports '''
import tensorflow as tf
import os


''' Limiting GPU usage on my PC '''
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

''' Imports for data cleaning '''
import cv2
import imghdr

''' Cleaning the data I got from google Images, and which I manually cleaned of vectors and very small files (<10KB) beforehand -> cheking for each picture in our data set if it is compatible withh opencv and if it matches our expected file format'''
data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path) 
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print ('Issue with image {}'.format(image_path))

'''Imports for loading the data'''
import numpy as np
from matplotlib import pyplot as plt

''' loading the dataset by creating a data pipeline using Keras Utils'''
data = tf.keras.utils.image_dataset_from_directory('data')
print("Klassen-Reihenfolge:", data.class_names)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

'''plotting the first 4 images from the first batch to identify the labels from the pipeline (Class 0 = Cat and Class 1 = Dog)'''          
fig,ax= plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
#plt.show()

'''preprocessing the data'''
'''scaling the images down from values between 0-255 to 0-1 inside of the data pipeline'''
data = data.map(lambda x,y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()
max = batch[0].max()
#print(f"{max}")


'''splitting the data into Training- / Validation- / Test-Sets ~50/33/17'''
batches = len(data)
print(f"{batches}")
train_size = int(len(data)*.7)-1
val_size = int(len(data)*.2)+1
'''so test_size != 0'''
test_size = int(len(data)*.1)+1 
print(f"{train_size}")
print(f"{val_size}")
print(f"{test_size}")
print(f"{train_size+val_size+test_size}")
'''making sure data is shuffeled before allocating'''
data = data.shuffle(buffer_size=1000) 
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


'''Building the Deep Learning model'''
'''imports'''
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

'''Building the Deep Learning model'''

'''Input Layer'''
inputs = tf.keras.Input(shape=(256, 256, 3), name='input_layer')

'''First Convolutional Block'''
x = tf.keras.layers.Conv2D(16, (3,3), strides=1, activation='relu', name='conv2d_1')(inputs)
x = tf.keras.layers.MaxPooling2D(name='maxpool_1')(x)

'''Second Convolutional Block'''  
x = tf.keras.layers.Conv2D(32, (3,3), strides=1, activation='relu', name='conv2d_2')(x)
x = tf.keras.layers.MaxPooling2D(name='maxpool_2')(x)

'''Third Convolutional Block'''
conv_features = tf.keras.layers.Conv2D(16, (3,3), strides=1, activation='relu', name='conv2d_3')(x)
x = tf.keras.layers.MaxPooling2D(name='maxpool_3')(conv_features)

'''Flatten and Dense Layers'''
x = tf.keras.layers.Flatten(name='flatten')(x)
x = tf.keras.layers.Dense(256, activation='relu', name='dense_1')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')(x)

'''Create the model'''
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='catdog_classifier')

'''Compiling'''
model.compile(
    optimizer='adam', 
    loss=tf.losses.BinaryCrossentropy(), 
    metrics=['accuracy']
)

'''model visualization'''
model.summary()

'''training the model'''
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
'''fitting the model'''
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

'''plotting the models performance'''
'''Loss'''
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
'''Accuracy'''
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")

plt.show()

'''Setting up the Evaluation of the Model'''
from keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
'''Evaluation based on the defined metrics'''
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(f"Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}")


'''Saving the model'''
from keras.models import load_model
model.save(os.path.join('models', 'catdogmodel.h5'))

