import keras
import cv2
import numpy as np
import tensorflow as tf

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19, preprocess_input

from model.define_model import dog_names, convert_to_label

breed_predictor = load_model('../model/Breed_Predictor.hd5f')

model_VGG19_features = VGG19(weights='imagenet', include_top=False)
ResNet50_model = ResNet50(weights='imagenet')

graph = tf.get_default_graph()


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))

    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def model_predict(img_path):
    global graph
    with graph.as_default():

        img_array = preprocess_input(path_to_tensor(img_path))

        VGG19_features = model_VGG19_features.predict(img_array)

        prediction = breed_predictor.predict(VGG19_features)

        breed = convert_to_label(prediction)

        return breed


def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = keras.applications.resnet50.preprocess_input(path_to_tensor(img_path))

    global graph
    with graph.as_default():
        prediction = np.argmax(ResNet50_model.predict(img))

    return prediction


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


def predict_breed(img_path):
    is_human = face_detector(img_path)
    is_dog = dog_detector(img_path)

    line_1 = "We think this image is of "
    line_2 = "We think this image mostly resembles a "
    if is_human & is_dog:
        title = line_1 + "either a dog or a human, we're not sure!\n" + "However... " + line_2
    elif is_human:
        title = line_1 + "a human!\n" + "However... " + line_2
    elif is_dog:
        title = line_1 + "a dog!\n" + line_2
    else:
        title = line_1 + "something other than a dog or a human!\n" + "However... " + line_2

    breed = ' '.join([word.capitalize() for word in model_predict(img_path).split('_')])

    return title + breed + '.'