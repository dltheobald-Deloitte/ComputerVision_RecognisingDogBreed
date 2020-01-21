import requests
import zipfile

def download_files(url, output_fp):
    """Downloads the files for the models which have been preprocessed and saved in
    the url provided.

    Parameters:
    url (String): The url where the features data is saved.
    output_fp (String): The location where the data should be saved
    """
    print('Retrieving data...')
    new_file = requests.get(url)
    print('Received.')

    print('Saving to output location...')
    open(output_fp, 'wb').write(new_file.content)
    print('Features saved')

#Defining where data is stored
url_VGG19 = r'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz'
url_VGG16 = r'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz'
url_dog_data = r'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip'
url_human_data = r'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip'

#Definign where to save this
destination_VGG19 = 'bottleneck_features/DogVGG19Data.npz'
destination_VGG16 = 'bottleneck_features/DogVGG16Data.npz'
destination_dog_data = '../dogImages.zip'
destination_human_data = '../lfw.zip'

#If this script is run as main, download the data above
if __name__ == '__main__':
    #Download all relevant files for the project
    download_files(url_VGG19,destination_VGG19)
    download_files(url_VGG16,destination_VGG16)
    download_files(url_dog_data,destination_dog_data)
    download_files(url_human_data,destination_human_data)

    #Unzip the files underneath
    zipfile.ZipFile(destination_dog_data,'r').extractall('..')
    zipfile.ZipFile(destination_human_data,'r').extractall('..')