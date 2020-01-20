import sys
import numpy as np

from glob import glob
from os.path import isfile

from keras.utils import np_utils
from keras.layers import Activation, GlobalAveragePooling2D, Activation, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image

from sklearn.datasets import load_files


# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("../../dogImages/train/*/"))]


def load_dataset(path_dict):
    preprocessed_features = r'../bottleneck_features/DogVGG19Data.npz'

    if isfile(preprocessed_features):
        bottleneck_features = np.load(preprocessed_features)
    else:
        print(r'Need to download bottleneck features at: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz')

    feature_sets = {}
    target_sets = {}

    for key in path_dict.keys():
        data = load_files(path_dict[key])

        dog_VGG19_features = bottleneck_features[key]
        dog_targets = np_utils.to_categorical(np.array(data['target']), 133)

        feature_sets[key] = dog_VGG19_features
        target_sets[key] = dog_targets

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


def convert_to_label(predicted_vector, breeds = dog_names):
    return breeds[np.argmax(predicted_vector)].split('.')[-1]


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
    
    pairings = [1 if convert_to_label(result[0]) == convert_to_label(result[1]) else 0 for result in zip(Y_pred,Y_test)]
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
    model.save(model_filepath)


def main():
    """ When this script is run as main, this creates a fitted model which can be used for predictions.
    It takes in specified system variables are used to exectue the machine learning pipeline:
        - loading in data
        - Builing the model
        - Evaluating the model
        - Saving the model
    """
    if len(sys.argv) == 2:
        model_filepath = sys.argv[1:][0]
        print('Loading data...')

        files = {'train' : '../../dogImages/train',
                'test' : '../../dogImages/test',
                'valid' : '../../dogImages/valid'}

        features, targets = load_dataset(files)
        
        print('Training model...')
        model = create_model_weights('best_weights_VGG19.hdf5', features['train'], targets['train'],
                                    features['valid'], targets['valid'])

        print('Evaluating model...')
        evaluate_model(model, features['test'], targets['test'])

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

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