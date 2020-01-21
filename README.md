# Udacity Data Scientist Nanodegree - nd025 - Project 3

### Requirements:
The python libraries required by this project are contained within 'requirements.txt'

## Project Definition
### Project Overview:
In this project, there's a preprocessing script to generate all relevant datasets which are available from: https://github.com/udacity/dog-project
This project is an application of CNNs and goes through the main steps of generating a CNN model using a simple neural network and escalating to transfer learning to predict the breed of a dog in an image. Moreover, this project also makes use of the ResNet50 model and face detection to identify whether an image contains a human or a dog. 

I have then taken the best approach of the above and used this to create a web-app and simple pipeline to classify images into breed of dog, also idetifying if the image is of a human or not. Moreover, I have created scripts which can be used to generate a model using the data sets given, although one has already been saved to this repository.

### Problem Statment
The problem which needs to be solved is clearly defined. A strategy for solving the problem, including discussion of the expected solution, has been made.

Images contain lots of information, not only in terms of pixels with a variety of red,green and blue. The information about space and where patterns in an image occurs can be informative and can be used to detect patterns for certain problems. The one which I have focussed on in this project is the classification of breeds of dog. In general, these images will be quite similar (i.e. 4 legs, 2 eyes, ears, nose, etc.), but we wish to detect nuances within the image itself which allows for more targeted detection.

The initial aim of this project is to detect the differences between humans and dogs, using a prebuilt ResNet50 model as well as a face_detection model. Next, we will focus on the dog portion in particular, creating a basic neural network from scratch and seeing what results we can yield. We will gradually add more components to thisand the expectation is to get over 60% accuracy in detecting dog breed.

### Metrics

The metric we will be using to evaluate this model will be accuracy. In terms of how the model performs, knowing how often the prediction is right will give us a good indication of what to expect from the model.

In terms of training, we will be using cross_entropy to decide what the best version of the model is. This is an accurate representation of how well the model is fitting as it maximises the probability that the solution is correctly guessing a validation set. It is a better choice over root mean squared error as it doesn't measure the distant (i.e. trying to obtain the probability 1), it instead maximises the classification over a wider range

## Analysis
### Data Exploration
Exploration of the data has been done in both of the dog_app.ipynb and Data_Exploration.ipynb notebooks. In short, I have looked at the performance of dog / human detection and noted that human face detection is not as good as the dog detection. The recall and precision of the dog_detector is 100%, whereas the recall of the face detect is 96% and the precision is 89%. Both of these are fine for our purposes as it is just an indication of whether an image is a human or a dog.

Further analysis has been done around the bias of the training set. By inspection, the images seem to be in only one orientation, so if images are at an angle, these may be missed by the training set. Moreover, the training set itself has a bias. There are more files for validation and training for some breeds over others, this means that there is a bigger incentive (i.e reduction in cross entropy loss) if certain breeds traits are targeted more. As a result this can cause overfitting and not generalise the model as well.


### Visualisations
Visualisations of the above are available in the Data_Exploration.ipynb notebook but can also be seen on the homepage of the app.
	
Features and calculated statistics relevant to the problem have been reported and discussed related to the dataset, and a thorough description of the input space or input data has been made. Abnormalities or characteristics about the data or input that need to be addressed have been identified.

## Methodology
### Data Pre-processing

Data has been preprocessed as outline within the dog_app.ipynb notebook. Images have been converted into 4D tensors and preprocessed using the VGG19 model without the final layers to give us the output features before predictions, otherwise known as transfer learning. These features are then used to predict a dog breed using a new model.
All of the features in the test/train/validation set have been preprocessed and have been downlaoded into the bottleneck_features folder, after following the setup instructions.

### Implementation & Refinement

This can be found in the dog_app.ipynb notebook, after each model has been built, it has been evaluated and next steps justified. The best/final solution was then made into a web-app.

## Instructions:

If you do not wish to follow the whole process and skip straight to the app, you can go straight to 2

1. Run the following commands in the project's root directory to set up the datasets required and set up the pre-requisites.

    - To download data (takes a while)
        `python Preprocessing.py`
    - To train a build, train, evaluate and save a new model, change into the model directory and run
        `python define_model.py Breed_Predictor.hdf5`
    - To get the necessary visualisation, change to the root folder and run the 'Data_Exploration.ipynb'

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000

### Files:
Within the root folder, the Preprocessing.py script and the Data_Exploration.ipynb are required to set up materials to run the app, if starting from scratch. This notebook, along with dog_app.ipynb, are the logical steps which talks through the building of the model and why I have taken certain steps.

Within the model folder, these are all the encessary scripts to build a new model. It also contains pre-run visualisation .pkl files and trained models.

The haarcascades folder contains an XML in which can be used to detect faces and bottleneckfeatures folder contaings the VGG19 preprocessed features.

Finally, within the app folder, there is a script run.py which can be used to start the flask app. Moreover, the static folder within here is the folder path used to upload files from the app. The templates are the html pages renders to display images.

## Results

The results have been referenced in the dog_app.ipynb folder, with the initial network yielding just over 4%, the VGG16 base model just over 40% and the VGG19 model with just over 70%.

Initially, the model used a simple user-defined architecture which would not have been complex enough to define patterns effectively. Since processing power and time were limited, transfer learning seemed like the obvious choice. This is where the features/patterns identified by other pre-defined algorithms could inform the model but would not be used directly for prediction.

The model which yielded the best results are the VGG19 method which used a combination of cross entropy and accuracy. Cross entropy as a metric would be minimised and therefore maximised the probability that the prediction was correct, whereas accuracy is a stand-alone metric which simply takes the predictions and compares them to the labels, and is definined independently of the training/valiation set. This makes accuracy the ideal metric to compare models.

## Conclusion
### Reflection
The solution starts with processing images by converting it into numpy arrays (i.e. 4D tensors) so that it can be understood by the algorithms. This is then fed into the input layer of the model and into subsequent dense/convolutional/pooling/activation layers to extract features from the image. Since the features needed are complex, it would take too long to train this from scratch.

As a result, we can take off the final layers from other neural networks which have been used to process images, in particular, VGG19. I found this particularly interesting as it allows you to leverage other peoples work and apply the same training that they have used to drastically reduce effort and the need for the same volume of training data.

Once a model is finalised and trained, the weights can be saved and, in essence, the model boils down to its base architecture and an array of numbers. I foudn this particularly interesting as well as this means that predictions can take place quickly and near-instantaneously, without takign up much memory.

Therefore, this model can be put into an app and deployed easily, which is what I have done. The app takes an image, processes it into a tensor, extracts the features from the VGG19 model and then uses a newly trained top layer to predict a dog breed. On top of this the solution also displays some information about the data and displays the original image.


### Improvement

I have put a lot of work into the app but I feel as though this could have been improved. Some additional coding, I could have made the image display dynamic so that anysize image would be adjusted to fit the screen. This would have given it a sleeker design and would have looked more professional.

Moreover, I could have made the pre-processing step more complex before feeding it into the model. I could have isolated different faces / dogs within an individual image and given a prediction to all of them. This would have provided more accuracy than a singular prediction for a group image and would have been more versatile.

Another improvement would be to set file extensions which could be ingested by the app. Currently, any extension could be uploaded but if it wasn't valid then the app will fail. Moreover, I could have given a layer of security tp the app by making sure the name was secure and wouldn't damage the app. This would have made it more robust and versatile again.