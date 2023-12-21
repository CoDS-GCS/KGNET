import requests
import sys
import os

import Constants

kgnet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
inference_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(kgnet_dir)
sys.path.append(inference_dir)
from Constants import *
def uploadModelToServer(model_path):
    with open(model_path,'rb') as f:
        # files = {'model' : (model_path,f)}
        files = {'model':  f}
        print('Uploading model to: ',Constants.KGNET_Config.GML_ModelManager_URL + ':' + Constants.KGNET_Config.GML_ModelManager_PORT+ '/uploadmodel/')
        response = requests.post(Constants.KGNET_Config.GML_ModelManager_URL + ':' + Constants.KGNET_Config.GML_ModelManager_PORT + '/uploadmodel/',
                                 files=files)
        if response.ok:
            print('Model uploaded to cloud successfully!')
        else:
            print('Error while uploading the model to cloud , error code: '+response)

def uploadDatasetToServer(dataset_path):
    with open(dataset_path,'rb') as f:
        # files = {'model' : (model_path,f)}
        files = {'dataset':  f}
        print('Uploading dataset to: ',Constants.KGNET_Config.GML_ModelManager_URL + ':' + Constants.KGNET_Config.GML_ModelManager_PORT+ '/uploaddataset/')
        response = requests.post(Constants.KGNET_Config.GML_ModelManager_URL + ':' + Constants.KGNET_Config.GML_ModelManager_PORT + '/uploaddataset/',
                                 files=files)
        if response.ok:
            print('Model uploaded to cloud successfully!')
        else:
            print('Error while uploading the model to cloud , error code: '+response)
def downloadModel (mid):
    response = requests.get(f"{Constants.KGNET_Config.GML_ModelManager_URL+':' + Constants.KGNET_Config.GML_ModelManager_PORT}/downloadmodel/{mid}", stream=True)
    if response.status_code != 500:
        filepath = os.path.join(Constants.KGNET_Config.trained_model_path, mid)
        with open(filepath,'wb') as f:
            for chunk in response.iter_content(chunk_size=128):
                f.write(chunk)
        return True

    else:
        print('File not found')
        return False

def downloadDataset (mid):
    response = requests.get(f"{Constants.KGNET_Config.GML_ModelManager_URL+':' + Constants.KGNET_Config.GML_ModelManager_PORT}/downloaddataset/{mid}", stream=True)
    if response.status_code != 500:
        filepath = os.path.join(Constants.KGNET_Config.inference_path, mid)

        with open(filepath,'wb') as f:
            for chunk in response.iter_content(chunk_size=128):
                f.write(chunk)
        print('Dataset Downloaded at ',filepath)
        return True

    else:
        print('File not found')
        return False
