import keras
import cv2
import numpy as np
import tensorflow as tf

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19, preprocess_input

from model.define_model import dog_names, convert_to_label

#Loading pre-built model for prediction
breed_predictor = load_model('../model/Breed_Predictor.hdf5')

#Loading VGG19 and resnet models to extract features and detect dogs
model_VGG19_features = VGG19(weights='imagenet', include_top=False)
ResNet50_model = ResNet50(weights='imagenet')

#Introducing default tensforflow graph for flask compatibility
graph = tf.get_default_graph()


def path_to_tensor(img_path):
    """Takes a file path of an image, loads it and then extracts information into a fixed shape numpy 
    array, ready to be used for VGG19 feature extracrion.

    Parameters:
    img_path (String): Path to a file containing the image.

    Returns:
    (np.array): A numpy array with fixed dimensions and information from an image
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))

    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def model_predict(img_path, breeds = dog_names):
    """Takes a file path of an image and returns a  dog breed prediction from this image.
    It does this by:
        1) Extracting and processing data from then image in the filepath
        2) Extracting the VGG19 features from the VGG19 model
        3) Uses a pre-built model to predict a dog breed category.
        4) Converts this output into a dog breed name.

    Parameters:
    img_path (String): Path to a file containing the image.
    breeds (List of String): Ordered list of dog breeds

    Returns:
    breed (String): A string containing the dog breed prediction.
    """
    #Makes the default graph for flask compatibility
    global graph
    with graph.as_default():

        #Generates a 4D tensor for the VGG19 model
        img_array = preprocess_input(path_to_tensor(img_path))

        #Generates the feauteres from the VGG19 model before prediction
        VGG19_features = model_VGG19_features.predict(img_array)

        #Predicts the breed of dog from features generated
        prediction = breed_predictor.predict(VGG19_features)

        #Converts breed prediction to dog breed label
        breed = convert_to_label(prediction, breeds)

        return breed


def face_detector(img_path):
    """Takes a file path of an image and detects whether a human face is present.

    Parameters:
    img_path (String): Path to a file containing the image.

    Returns:
    (Boolean): Presence of a human face
    """
    #Loads pre-built face detector
    face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')

    #Converts image into usable array
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Produces list of faces detected
    faces = face_cascade.detectMultiScale(gray)

    return len(faces) > 0


def ResNet50_predict_labels(img_path):
    """Takes a file path of an image and classifies the image into different categories.

    Parameters:
    img_path (String): Path to a file containing the image.

    Returns:
    prediction (Integer): A number indicating the classification of the image
    """
    # returns vector for image located at img_path
    img = keras.applications.resnet50.preprocess_input(path_to_tensor(img_path))

    #Classifies the image into a category
    global graph
    with graph.as_default():
        prediction = np.argmax(ResNet50_model.predict(img))

    return prediction


def dog_detector(img_path):
    """Takes a file path of an image and returns whether it is a dog or not

    Parameters:
    img_path (String): Path to a file containing the image.

    Returns:
    (Boolean): Presence of a dog
    """
    #predicts the category of an image
    prediction = ResNet50_predict_labels(img_path)

    return ((prediction <= 268) & (prediction >= 151)) 


def predict_breed(img_path):
    """Takes a file path of an image, classifies if it is potentially a dog or human then
    states what breed the image mostly resembles.

    Parameters:
    img_path (String): Path to a file containing the image.

    Returns:
    (Boolean): The breed which the image submitted most resembles
    """
    #Detects if an image contains a dog or a human
    is_human = face_detector(img_path)
    is_dog = dog_detector(img_path)

    #Instatiates statement structures
    line_1 = "We think this image is of "
    line_2 = "We think this image mostly resembles a "

    #Returns a statement based on dog/human classifications
    if is_human & is_dog:
        title = line_1 + "either a dog or a human, we're not sure!\n" + "However... " + line_2
    elif is_human:
        title = line_1 + "a human!\n" + "However... " + line_2
    elif is_dog:
        title = line_1 + "a dog!\n" + line_2
    else:
        title = line_1 + "something other than a dog or a human!\n" + "However... " + line_2

    #Reformats breed prediction and appends this onto the statement
    breed = ' '.join([word.capitalize() for word in model_predict(img_path).split('_')])

    return title + breed + '.'