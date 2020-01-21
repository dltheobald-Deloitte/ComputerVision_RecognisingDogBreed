import os
import sys
import json
import pickle

import plotly
from plotly import graph_objs as go

from flask import Flask
from flask import render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES

#Adding upper folder to path to import breed predictor
sys.path.append('..')
from model.prediction import predict_breed

#Define flask app to run
app = Flask(__name__)

#Defining an upload folder and configuring this in flask app
photos = UploadSet('photos', IMAGES)
upload_folder = 'static'
app.config['UPLOADED_PHOTOS_DEST'] = upload_folder
configure_uploads(app,photos)

#Loading pre-saved data for graphs
df_valid = pickle.load(open('../model/Breed_Counts_valid.pkl','rb'))
df_train = pickle.load(open('../model/Breed_Counts_trin.pkl','rb'))
human_vals, dog_vals = pickle.load(open('../model/BarChart_1.pkl','rb'))

#Defining the homepage of the app being run.
@app.route('/')
@app.route('/index')
def index():
    """ Renders a html template with graphs and gives an option to upload a file into the folder
    define above.
    """
    #Instatiates list of plotly objects used in the html template
    graphs = []

    #Defines the first Bar graph and its layout for the homepage
    data_valid = go.Bar(x = list(df_valid.Breed), y=list(df_valid.train_count))
    title = 'Bias/Counts of Validation Images'
    x_label = 'Dog Breed'
    y_label = 'Count'

    graph_1 = {'data': [data_valid],
            'layout': {
                'title': title,
                'yaxis': {'title': x_label},
                'xaxis': {'title': y_label}
            }}

    graphs.append(graph_1)

    #Defines the first Bar graph and its layout for the homepage
    data_train = go.Bar(x = list(df_train.Breed), y=list(df_train.train_count))
    title = 'Bias/Counts of Training Images'
    x_label = 'Dog Breed'
    y_label = 'Count'

    graph_2 = {'data': [data_train],
            'layout': {
                'title': title,
                'yaxis': {'title': x_label},
                'xaxis': {'title': y_label}
            }}

    graphs.append(graph_2)

    #Add a second graph (grouped bar chart) and updates its layout
    detected = ['Humans', 'Dog']

    graph_3 = go.Figure(data=[
    go.Bar(name='Human_face_detector', x=detected, y=human_vals),
    go.Bar(name='Dog_face_detector', x=detected, y=dog_vals)
    ])
    
    graph_2.update_layout(barmode='group', showlegend = True, 
                        title="Accuracy for a sample\nby Group and Detector type",
                        yaxis_title=r"% of samples")

    graphs.append(graph_3)

    #Adds id tags to the graphs for reference in html 
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]

    #Converts graphs to Json formar for rendering to a template.
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    """ When submitting an image, this function will save the file into the upload folder
    and predicts what breed of dog would be contained in the image.

    This function will also render a template to displayh the results of the prediction.

    If no image is submitted, this will redirect the user back to the homepage
    """
    #When an image is submitted to the URL '/upload_image'
    if request.method == 'POST' and 'photo' in request.files:

        #Save the file into the upload_folder
        filename = photos.save(request.files['photo'])

        #Predicts what breed the photo contains
        saved_img_path = os.path.join(sys.path[0], upload_folder, filename)
        prediction = predict_breed(saved_img_path)

        #Renders template to display results
        return render_template(
                'upload_image.html',
                img_name=filename,
                prediction=prediction
                )

    #If no image submitted, redirects to homepage
    return redirect(url_for('index'))


def main():
    """When this script is run as the main script, it runs a flask app on:
        http://127.0.0.1:5000
    """
    app.run(host = '127.0.0.1', port =5000, debug = True)


#When this script is run as main, runs the app.
if __name__ == '__main__':
   main()