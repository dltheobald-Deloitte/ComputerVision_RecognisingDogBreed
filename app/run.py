import os
import sys
import json
import pickle

import plotly
from plotly import graph_objs as go

from flask import Flask
from flask import render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES
print(sys.path)
sys.path.append('..')
from model.prediction import predict_breed
print(sys.path)

app = Flask(__name__)

photos = UploadSet('photos', IMAGES)

upload_folder = 'static'
#os.path.join(os.getcwd(), '..', 'samples')
# #os.path.join('..','samples')

app.config['UPLOADED_PHOTOS_DEST'] = upload_folder

configure_uploads(app,photos)

df = pickle.load(open('../model/Breed_Counts.pkl','rb'))
human_vals, dog_vals = pickle.load(open('../model/BarChart_1.pkl','rb'))

@app.route('/')
@app.route('/index')
def index():
    graphs = []

    data = go.Bar(x = list(df.Breed), y=list(df.train_count))
    title = 'Margins for count of Training Images'
    x_label = 'Dog Breed'
    y_label = 'Count'

    graph_1 = {'data': [data],
            'layout': {
                'title': title,
                'yaxis': {'title': x_label},
                'xaxis': {'title': y_label}
            }}

    graphs.append(graph_1)

    detected = ['Humans', 'Dog']

    graph_2 = go.Figure(data=[
    go.Bar(name='Human_face_detector', x=detected, y=human_vals),
    go.Bar(name='Dog_face_detector', x=detected, y=dog_vals)
    ])
    # Change the bar mode
    graph_2.update_layout(barmode='group', showlegend = True, 
                        title="Accuracy for a sample\nby Group and Detector type",
                        yaxis_title=r"% of samples")

    graphs.append(graph_2)

    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]

    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        saved_img_path = os.path.join(sys.path[0], upload_folder, filename)
        prediction = predict_breed(saved_img_path)
        return render_template(
                'upload_image.html',
                img_name=filename,
                prediction=prediction
                )

    return redirect(url_for('index'))#'upload_image.html')


def main():
   app.run()
    #app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
   main()