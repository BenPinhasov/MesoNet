import numpy as np
from classifiers import Meso4
# from pipeline import *
import cv2
import dlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.preprocessing import normalize
import tensorflow as tf


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF.h5')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

# dataGenerator = ImageDataGenerator(rescale=1./255)
# generator = dataGenerator.flow_from_directory(
#         'test_images',
#         target_size=(256, 256),
#         batch_size=1,
#         class_mode='binary',
#         subset='training')
#
# # 3 - Predict
# X, y = generator.next()
video_path = 'test_images/real/000.mp4'
reader = cv2.VideoCapture(video_path)
while reader.isOpened():
    _, image = reader.read()
    if image is None:
        break
    height, width = image.shape[:2]

    face_detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    if len(faces):
        # For now only take biggest face
        face = faces[0]

        # --- Prediction ---------------------------------------------------
        # Face crop with dlib and bounding box scale enlargement
        x, y, size = get_boundingbox(face, width, height)
        cropped_face = image[y:y + size, x:x + size]
        # resize to 256, 256
        cropped_face = cv2.resize(cropped_face, (256, 256))
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        # Convert to numpy array
        img = img_to_array(cropped_face)

        # Normalize image
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        img = (img - mean) / std

        # Permute dimensions from HWC to CHW
        # img = img.transpose(2, 0, 1)

        # Convert to tensor

        img = tf.convert_to_tensor(img)

        # Add batch dimension
        img = tf.expand_dims(img, 0)
        classification = classifier.predict(img)
        print('Predicted :', classification, '\nReal class :', 0)
