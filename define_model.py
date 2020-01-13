import sys
import pickle
import numpy as np

from glob import glob
from os.path import isfile

from keras.utils import np_utils
from keras.layers import Activation, GlobalAveragePooling2D, Flatten, Activation, Dense, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
#from keras.applications.vgg19 import VGG19, preprocess_input
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.preprocessing import FunctionTransformer
from .extract_bottleneck_features import extract_VGG19

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("../dogImages/train/*/"))]


def load_dataset(path_dict):
    preprocessed_features = r'bottleneck_features/DogVGG19Data.npz'

    if isfile(preprocessed_features):
        bottleneck_features = np.load('bottleneck_features/DogVGG19Data.npz')
    else:
        print(r'Need to download bottleneck features at: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz')

    feature_sets = {}
    target_sets = {}

    for key in path_dict.keys():
        data = load_files(path_dict[key])

        dog_VGG19_features = bottleneck_features[key]
        dog_targets = np_utils.to_categorical(np.array(data['target']), 133)

        feature_sets[key] = dog_target
        target_sets[key] = dog_target

    return feature_sets, target_sets


def define_model(train_data):
    #Instatiating model
    model = Sequential()

    #Adding layer to decrease dimensions and reclassify model
    model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))

    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(133))
    model.add(Activation('softmax'))

    #Prints the architecture of the model
    model.summary()

    #Defines the loss function, optimizer and the metric to measure model performance
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def create_model_weights(output_path, X_train, y_train, X_validation, y_validation):
    model = define_model(X_train)

    checkpointer = ModelCheckpoint(filepath=output_path,
                                verbose=1, save_best_only=True)

    model.fit(X_train, y_train, 
            validation_data=(X_validation, y_validation),
                epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

    return model


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


#def extract_VGG19(tensor):
#	return VGG19(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def build_pipeline(model):
    """ Creates an instance of model along with a machine learning pipeline which, when fit,
    will process the features data and select the best parameters from those given to improve the model outputs.

    Returns:
    cv (GridSearchCV object): A machine learning pipeline and parameters to configure a model and process
                              its features 
    """
    #Defines pipeline to calculate the tfidf of each message and pass through the categories features.
    #This pipeline will also trains the classifier

    ##bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
    ###NEED TO ADD PREREQUISITES TO THIS, I.E. THE ABOVE AND THE DOG TRAIN/HUMAN TRAIN FILES.
    pipeline = Pipeline([ 
        ('img_array', FunctionTransformer(path_to_tensor, validate=False)]),
        ('preprocess', FunctionTransformer(extract_VGG19, validate=False)),
        ('clf', KerasClassifier(model))
        ])

    return pipeline


def convert_to_label(predicted_vector, dog_names = dog_names):
    return dog_names[np.argmax(predicted_vector)].split('.')[-1]


def evaluate_model(model, X_test, Y_test):
    """ Evaluates and returns the models performance with respect to precision, recall and f1_score

    Parameters:
    model (GridSearchCV object): A fitted model which can be used to predictr results 
    X_test (pd.DataFrame): A dataframe with the test set of data to transform into features.
    Y_test (pd.DataFrame): A dataframe with the test set of data labels.
    category_names (list of String): A list with the category/label description
    """
    #Prediction outputs from fitted model
    Y_pred = model.predict(X_test)

    pairings = [1 for result in zip (Y_pred,Y_test) if convert_to_label(result[0]) = result[1].split('.')[-1] else 0]
    accuracy = float(sum(pairings))/len(pairings)
    
    #Comparing predictions to actuals and printing metrics
    print('The model accurately predicts the dog breed ' + "{0:.0%}".format(accuracy) + ' of the time.')


def save_model(model, model_filepath):
    """ Saves a trained version of the model in the location specified in model_filepath

    Parameters:
    model (GridSearchCV object): A fitted model which can be used to predict results 
    model_filepath (String): The location of where the model shoule be saved
    """
    #Saving a copy of the fitted model
    pickle.dump(model, open(model_filepath, 'wb+'))


def main():
    """ When this script is run as main, this creates a fitted model which can be used for precictions.
    It takes in specified system variables are used to exectue the machine learning pipeline:
        - loading in data
        - Builing the model
        - Evaluating the model
        - Saving the model
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        files = {'train' : '../dogImages/train',
                'test' : '../dogImages/valid',
                'valid' : '../dogImages/test'}

        features, targets = load_dataset(files)
        
        print('Training model...')
        model_keras = create_model_weights('model/best_weights.hd5f',features['train'], targets['train'],
                                    features['valid'], targets['valid'])
                                    
        model_sklearn = build_pipeline(model_keras)

        print('Evaluating model...')
        evaluate_model(model_keras, features['test'], targets['test'])

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model_sklearn, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()




###################################################
#from keras import backend as K
## with a Sequential model, give yu certain output
#get_3rd_layer_output = K.function([model.layers[0].input],
#                                  [model.layers[3].output])
#layer_output = get_3rd_layer_output([x])[0]
#####################################################