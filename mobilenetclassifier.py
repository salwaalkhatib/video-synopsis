from unicodedata import category
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications import MobileNet, imagenet_utils
from keras import Input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import utils
import os

def mobilenet_color_model(input_size, output_size, weights="imagenet"):
    """Creates model architecture for the color classifier

    Args:
        input_size (tuple): 3-element tuple that represents image input size
        output_size (int): number of classes
        weights (string): path to weights file

    Returns:
        keras.models.Model: mobilenet network
    """    
    base_model = MobileNet(include_top = False, input_shape = input_size, weights="imagenet")
    # Freeze base model
    base_model.trainable = False
    inputs = Input(shape=input_size)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    dense3 = Dense(512, activation = 'relu')(x)
    dropout2 = Dropout(0.2)(dense3)
    dense4 = Dense(512, activation = 'relu')(dropout2)
    dropout3 = Dropout(0.2)(dense4)
    dense5 = Dense(128, activation = 'relu')(dropout3)
    dropout4 = Dropout(0.2)(dense5)
    outputs = Dense(output_size, activation = 'softmax')(dropout4)
    model = Model(inputs, outputs)
    if weights is not None:
        model.load_weights(weights)
    return model

def mobilenet_type_model(input_size, output_size, weights=None):
    """Creates model architecture for the type classifier

    Args:
        input_size (tuple): 3-element tuple that represents image input size
        output_size (int): number of classes
        weights (string): path to weights file

    Returns:
        keras.models.Model: mobilenet network
    """    
    base_model = MobileNet(include_top = False, input_shape = input_size, weights=None)
    # Freeze base model
    base_model.trainable = False
    inputs = Input(shape=input_size)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    dense1 = Dense(1024, activation = 'relu')(x)
    dropout = Dropout(0.3)(dense1)
    dense2 = Dense(1024, activation = 'relu')(dropout)
    dropout1 = Dropout(0.3)(dense2)
    dense3 = Dense(512, activation = 'relu')(dropout1)
    dropout2 = Dropout(0.2)(dense3)
    dense4 = Dense(512, activation = 'relu')(dropout2)
    dropout3 = Dropout(0.1)(dense4)
    dense5 = Dense(128, activation = 'relu')(dropout3)
    dropout4 = Dropout(0.1)(dense5)
    outputs = Dense(output_size, activation = 'softmax')(dropout4)
    model = Model(inputs, outputs)
    if weights is not None:
        model.load_weights(weights)
    return model

# model = Sequential()
# base_model = MobileNet(include_top = False, input_shape = (224,224,3), weights="imagenet")
# # Freeze base model
# base_model.trainable = False

# inputs = Input(shape=(224, 224, 3))
# x = base_model(inputs)
# x = GlobalAveragePooling2D()(x)
# dense1 = Dense(1024, activation = 'relu')(x)
# dropout = Dropout(0.3)(dense1)
# dense2 = Dense(1024, activation = 'relu')(dropout)
# dropout1 = Dropout(0.3)(dense2)
# dense3 = Dense(512, activation = 'relu')(dropout1)
# dropout2 = Dropout(0.2)(dense3)
# dense4 = Dense(512, activation = 'relu')(dropout2)
# dropout3 = Dropout(0.2)(dense4)
# dense5 = Dense(128, activation = 'relu')(dropout3)
# dropout4 = Dropout(0.2)(dense5)
# outputs = Dense(6, activation = 'softmax')(dropout4)

# inputs = Input(shape=(224, 224, 3))
# x = base_model(inputs)
# x = GlobalAveragePooling2D()(x)
# dense1 = Dense(1024, activation = 'relu')(x)
# dropout = Dropout(0.3)(dense1)
# dense2 = Dense(1024, activation = 'relu')(dropout)
# dropout1 = Dropout(0.3)(dense2)
# dense3 = Dense(512, activation = 'relu')(dropout1)
# dropout2 = Dropout(0.2)(dense3)
# dense4 = Dense(512, activation = 'relu')(dropout2)
# dropout3 = Dropout(0.1)(dense4)
# dense5 = Dense(128, activation = 'relu')(dropout3)
# dropout4 = Dropout(0.1)(dense5)
# outputs = Dense(5, activation = 'softmax')(dropout4)

# model = Model(inputs, outputs)

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.1,
#         width_shift_range=0.2, 
#         height_shift_range=0.2,
#         brightness_range=[0.2,0.8],
#         horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# training_set = train_datagen.flow_from_directory(
#             'D:/VeRi/VeRi/cleaned type/image_train_type/',
#             target_size=(224, 224),
#             batch_size=32,
#             class_mode='categorical',
#             shuffle=True)

# test_set = test_datagen.flow_from_directory(
#             'D:/VeRi/VeRi/cleaned type/image_test_type/',
#             target_size=(224, 224),
#             batch_size=32,
#             class_mode='categorical',
#             shuffle=True)

# checkpoint = ModelCheckpoint("type-classification-model/mobilenet_cleaned_data_10epoch_best.hdf5", monitor='val_acc', verbose=1,
#     save_best_only=True, mode='auto', period=1)

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# test_steps_per_epoch = np.math.ceil(test_set.samples / test_set.batch_size)
# train_steps_per_epoch = np.math.ceil(training_set.samples / training_set.batch_size)
# model.load_weights('type-classification-model/mobilenet_cleaned_data_5epoch_best.hdf5')
# model.fit_generator(training_set, epochs=5, validation_data=test_set, steps_per_epoch=train_steps_per_epoch, validation_steps=test_steps_per_epoch, callbacks=[checkpoint])
# model.save('type-classification-model/mobilenet_cleaned_data_10epoch_final.hdf5')


# # Confusion matrix
# import sklearn.metrics as metrics
# model.load_weights('type-classification-model/mobilenet_cleaned_data_10epoch_best.hdf5')
# predictions = model.predict_generator(test_set, steps=test_steps_per_epoch)
# # Get most likely class
# predicted_classes = np.argmax(predictions, axis=1)
# true_classes = test_set.classes
# class_labels = list(test_set.class_indices.keys())
# report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
# #open text file
# text_file = open("type-classification-model/classification-report_mobilenet_cleaned_data_10epoch.txt", "w")
# #write string to file
# text_file.write(report)
# #close file
# text_file.close()

#Testing per image
#model.load_weights('color-classification-model/224x224 - 6 classes - smaller model/mobilenet_cleaned_data_10epoch_best.hdf5')
# make predictions on test image using mobilenet
# image = utils.preprocess_image_mobilenet('0135_c014_00058300_1.jpg', (224,224))
# prediction = model.predict(image)

# frames=[]
# for file in os.listdir('D:/Video Synopsis Tool/predictions/'):
#     if(file.endswith('.jpg')):
#         frames.append(utils.preprocess_image_mobilenet('D:/Video Synopsis Tool/predictions/' + file, (224,224)))

# model = mobilenet_model((224,224,3), 6, 'color-classification-model/224x224 - 6 classes - smaller model/mobilenet_cleaned_data_10epoch_best.hdf5')
# prediction = model.predict(np.vstack(frames))
# print(prediction)
# decoded = utils.decode_mobilenet_predictions(prediction, top=2)
# print(decoded)

# for index, image in enumerate(os.listdir('D:/Video Synopsis Tool/predictions/')):
#     if(image.endswith(".png") or image.endswith(".jpg")):
#         image = cv2.imread('D:/Video Synopsis Tool/predictions/' + image)
#         org = (int(image.shape[0]/2), int(image.shape[1]/2))

#         # setup text
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         text = decoded[index][0]
#         # get boundary of this text
#         textsize = cv2.getTextSize(text, font, 1, 2)[0]
#         # get coords based on boundary
#         textX = int((image.shape[1] - textsize[0]) / 2)
#         textY = int((image.shape[0] + textsize[1]) / 2)

#         cv2.putText(img=image, text=text, org=(textX, textY), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0),thickness=3)
#         cv2.imwrite(f'D:/Video Synopsis Tool/predictions/pred/{index}.png', image)

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    sampleClassificationReport = """              precision    recall  f1-score   support

        black       0.67      0.73      0.69      3884
            blue       0.33      0.29      0.31       230
            gray       0.40      0.40      0.40      1822
            red       0.31      0.34      0.32      1099
        white       0.64      0.68      0.66      2381
        yellow       0.29      0.27      0.28       727

        avg/total       0.54    0.57        0.55    10143    
    """

    utils.plot_classification_report(sampleClassificationReport)
    plt.show()
    plt.close()