import numpy as np
from classifiers import *
# from pipeline import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF.h5')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

dataGenerator = ImageDataGenerator(rescale=1. / 255)
generator = dataGenerator.flow_from_directory(
    'test_images',
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary',
    subset='training')

# 3 - Predict
data_list = []
batch_index = 0
while batch_index <= generator.batch_index:
    data = generator.next()
    data_list.append(data[0])
    X, y = data
    batch_index = batch_index + 1
    print('Predicted :', classifier.predict(X), '\nReal class :', list(generator.class_indices.keys())[int(y[0])])
    # X, y = generator.next()

pass
# # 4 - Prediction for a video dataset
#
# classifier.load('weights/Meso4_F2F.h5')
#
# predictions = compute_accuracy(classifier, 'test_videos')
# for video_name in predictions:
#     print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
