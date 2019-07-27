# -*- coding: utf-8 -*-''
"""
Created on Mon Apr 22 19:30:10 2019

@author: shubh
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_yaml


classifier=Sequential()


classifier.add(Convolution2D(64, 3, 3,input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Convolution2D( 64, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Flatten())


classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))



classifier.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['accuracy'])
classifier.load_weights("weights.02-0.49.hdf5")



from keras.callbacks import EarlyStopping, ReduceLROnPlateau
#earlystop = EarlyStopping(patience=10)
#learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
earlystop=EarlyStopping(monitor='val_acc', min_delta=0, patience=0, verbose=0, mode='max', baseline=None, restore_best_weights=False)



from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)


train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


from keras.callbacks import ModelCheckpoint
import keras.callbacks as kc
filepath = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint=ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='min', period=1)
tensorboard=kc.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
classifier.fit_generator(train_set,
                         steps_per_epoch=8000,
                         epochs=20,
                         validation_data=test_set,
                         validation_steps=2000,
                         callbacks = [checkpoint,tensorboard,earlystop])




#prediction of results
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('cat.jpg', target_size=(64, 64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)
result=classifier.predict(test_image)
print(result)
train_set.class_indices

classifier.summary()
classifier.save('model1.h5')

#
#model=classifier
#model_yaml = model.to_yaml()
#with open("model.yaml", "w") as yaml_file:
#yaml_file.write(model_yaml)
# serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk") 
# later... 
# load YAML and create model
#yaml_file = open('model.yaml', 'r')
#loaded_model_yaml = yaml_file.read()
#yaml_file.close()
#loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk") 
# evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.predict(test_image)
#print(score)