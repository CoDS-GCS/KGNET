import json
import os.path

import pandas as pd
import subprocess
import argparse
import sys

import Constants

sys.path.append('..')
from GMLaaS.DataTransform.TSV_TO_PYG_dataset import transform_tsv_to_PYG
from GMLaaS.DataTransform.Transform_LP_Dataset import transform_LP_train_valid_test_subsets
from GMLaaS.models.graph_saint_KGTOSA import graphSaint
from GMLaaS.models.rgcn_KGTOSA import rgcn
from GMLaaS.models.graph_saint_Shadow_KGTOSA import graphShadowSaint
from GMLaaS.models.graph_MorsE import run_morse
from GMLaaS.models.rgcn.rgcn_link_pred import rgcn_lp
from GMLaaS.model_manager import uploadModelToServer, uploadDatasetToServer,uploadEmbToServer
from GMLaaS.models.evaluater import Evaluator
from Constants import *


def load_args(path_json):
    with open(path_json) as json_file:
        json_args = json.load(json_file)
    return json_args


def format_args(task, json_args, path_script):
    format_args = []
    format_args.extend(['python', path_script])
    for arg in json_args[task]:
        string_arg = '--' + str(arg) + '=' + str(json_args[task][arg])
        format_args.append(string_arg)
    return format_args


# data['transformation_args']
# list_transformation = []
# list_transformation.extend(['python','Transformation/TSV_TO_PYG_dataset.py'])
# # list_transformation.append('Transformation/TSV_TO_PYG_dataset.py')
# for arg in data['transformation_args'].keys():
#     format_arg = '--'+str(arg)+'='+str(data['transformation_args'][arg])
#     list_transformation.append(format_arg)

# subprocess.run(list_transformation)

# list_training = []
# list_training.extend(['python','models/models/graph_saint/graph_saint_KGTOSA.py'])
# # list_training.append('models/models/graph_saint/graph_saint_KGTOSA.py')
# for arg in data['training_args'].keys():
#     format_arg = '--'+str(arg)+'='+str(data['training_args'][arg])
#     list_training.append(format_arg)

# subprocess.run(list_training)

def uploadModel(model_path):
    try:
        emb_path = os.path.join(KGNET_Config.emb_store_path,model_path+'_wise.zip')
        if Constants.KGNET_Config.fileStorageType==Constants.FileStorageType.remoteFileStore:
            uploadModelToServer(os.path.join(Constants.KGNET_Config.trained_model_path,model_path + '.model'))
            uploadModelToServer(os.path.join(Constants.KGNET_Config.trained_model_path,model_path + '.param'))
            if os.path.exists(emb_path):
                print('Uploading WISE version..')
                uploadEmbToServer(emb_path)
                uploadModelToServer(os.path.join(Constants.KGNET_Config.trained_model_path,model_path + '_wise.model'))
                uploadModelToServer(os.path.join(Constants.KGNET_Config.trained_model_path, model_path + '_wise.param'))

        elif Constants.KGNET_Config.fileStorageType == Constants.FileStorageType.S3:
            utils.uploadFileToS3(os.path.join(Constants.KGNET_Config.trained_model_path,model_path) ,file_type="model")
            utils.uploadFileToS3(os.path.join(Constants.KGNET_Config.trained_model_path,model_path + '.param'),file_type="metadata")
    except Exception as e:
        print("uploadModel Failed:",e)


def cmd_run_training_pipeline(path_json=None, path_transformation_py='DataTransform/TSV_TO_PYG_dataset.py',
                              path_training_py='models/models/graph_saint/graph_saint_KGTOSA.py', json_args=None):
    if json_args is None:
        json_args = load_args(path_json)
    else:
        json_args = json_args
    args_transformation = format_args(task='transformation', json_args=json_args,
                                      path_script=path_transformation_py)
    args_training = format_args(task='training', json_args=json_args, path_script=path_training_py)
    subprocess.run(args_transformation)
    subprocess.run(args_training)


def run_training_pipeline(json_args):
    print("################# Start GMLaaS Pipline ###########################")
    print("######### Start PYG Dataset Transformation ##########")
    if json_args["transformation"]["operatorType"] == Constants.GML_Operator_Types.NodeClassification:
        transform_results_dict = transform_tsv_to_PYG(dataset_name=json_args["transformation"]["dataset_name"],
                                                      dataset_name_csv=json_args["transformation"]["dataset_name"],
                                                      dataset_types=json_args["transformation"]["dataset_types"],
                                                      split_rel="random",
                                                      target_rel=json_args["transformation"]["target_rel"],
                                                      similar_target_rels=[],
                                                      output_root_path=json_args["transformation"]["output_root_path"],
                                                      MINIMUM_INSTANCE_THRESHOLD=json_args["transformation"]["MINIMUM_INSTANCE_THRESHOLD"],
                                                      test_size=json_args["transformation"]["test_size"],
                                                      valid_size=json_args["transformation"]["valid_size"],
                                                      split_rel_train_value=None,
                                                      split_rel_valid_value=None,
                                                      labelNodetype=json_args["transformation"]["label_node_type"],
                                                      targetNodeType=json_args["transformation"]["target_node_type"])
        print("Task ClassesCount:",transform_results_dict["ClassesCount"])
        if json_args["training"]["GNN_Method"] == Constants.GNN_Methods.Graph_SAINT:
            train_results_dict = graphSaint(device=0, num_layers=2, hidden_channels=64, dropout=0.5, lr=0.005,
                                            epochs=(json_args["training"]["epochs"] if "epochs" in json_args["training"].keys() else 20),
                                            runs=1, batch_size=20000,
                                            walk_length=2, num_steps=10, loadTrainedModel=0,
                                            dataset_name=json_args["training"]["dataset_name"],
                                            root_path=json_args["training"]["root_path"],
                                            output_path=json_args["training"]["root_path"],
                                            include_reverse_edge=True,
                                            n_classes=transform_results_dict["ClassesCount"],
                                            emb_size=(json_args["training"]["embSize"] if "embSize" in json_args["training"].keys() else 128),
                                            label_mapping=transform_results_dict['label_mapping']
                                            )
            uploadModel(train_results_dict['model_name'])

        elif json_args["training"]["GNN_Method"] == Constants.GNN_Methods.RGCN:
            train_results_dict = rgcn(device=0, num_layers=2, hidden_channels=64, dropout=0.5, lr=0.005,
                                      epochs=(json_args["training"]["epochs"] if "epochs" in json_args["training"].keys() else 20),
                                      runs=1, batch_size=20000,
                                      walk_length=2, num_steps=10, loadTrainedModel=0,
                                      dataset_name=json_args["training"]["dataset_name"],
                                      root_path=json_args["training"]["root_path"],
                                      output_path=json_args["training"]["root_path"],
                                      include_reverse_edge=True,
                                      n_classes=transform_results_dict["ClassesCount"],
                                      emb_size=(json_args["training"]["embSize"] if "embSize" in json_args["training"].keys() else 128))
            uploadModel(train_results_dict['model_name'])

        elif json_args["training"]["GNN_Method"] == Constants.GNN_Methods.ShaDowGNN:
            train_results_dict = graphShadowSaint(device=0, num_layers=2, hidden_channels=64, dropout=0.5, lr=0.005,
                                                  epochs= (json_args["training"]["epochs"] if "epochs" in json_args["training"].keys() else 20),
                                                  runs=1, batch_size=20000,
                                                  walk_length=2, num_steps=10, loadTrainedModel=0,
                                                  dataset_name=json_args["training"]["dataset_name"],
                                                  root_path=json_args["training"]["root_path"],
                                                  output_path=json_args["training"]["root_path"],
                                                  include_reverse_edge=True,
                                                  n_classes=transform_results_dict["ClassesCount"],
                                                  emb_size=(json_args["training"]["embSize"] if "embSize" in json_args["training"].keys() else 128),
                                                  label_mapping=transform_results_dict['label_mapping'])
            uploadModel(train_results_dict['model_name'])

    elif json_args["transformation"]["operatorType"] == Constants.GML_Operator_Types.LinkPrediction:
        transform_results_dict = transform_LP_train_valid_test_subsets(
            data_path=json_args["transformation"]["output_root_path"],
            ds_name=json_args["transformation"]["dataset_name"],
            target_rel=json_args["transformation"]["target_rel"],
            valid_size=0.1, test_size=0.1,
            delm='\t', containHeader=False)
        if json_args["training"]["GNN_Method"] == Constants.GNN_Methods.MorsE:
            train_results_dict = run_morse(dataset_name=json_args["training"]["dataset_name"],
                                           root_path=json_args["training"]["root_path"],
                                           epochs=(json_args["training"]["epochs"] if "epochs" in json_args["training"].keys() else 5),
                                           embSize=(json_args["training"]["embSize"] if "embSize" in json_args["training"].keys() else 128))

        elif json_args['training']['GNN_Method'] == Constants.GNN_Methods.RGCN:
            train_results_dict = rgcn_lp(dataset_name=json_args["training"]["dataset_name"],
                                         root_path=json_args["training"]["root_path"],
                                         epochs=(json_args["training"]["epochs"] if "epochs" in json_args["training"].keys() else 5),
                                         hidden_channels=(json_args["training"]["embSize"] if "embSize" in json_args["training"].keys() else 10))
        uploadModel(train_results_dict['model_name'])

    # UPLOAD DATASET


    if json_args["transformation"]["operatorType"] == Constants.GML_Operator_Types.NodeClassification:
        if Constants.KGNET_Config.fileStorageType == Constants.FileStorageType.remoteFileStore:
            uploadDatasetToServer(os.path.join(Constants.KGNET_Config.datasets_output_path, json_args["training"]["dataset_name"] + '.zip'))
        elif Constants.KGNET_Config.fileStorageType == Constants.FileStorageType.S3:
            utils.uploadFileToS3(os.path.join(Constants.KGNET_Config.datasets_output_path, json_args["training"]["dataset_name"] + '.zip'), file_type="metadata")
    elif json_args["transformation"]["operatorType"] == Constants.GML_Operator_Types.LinkPrediction:
        if Constants.KGNET_Config.fileStorageType == Constants.FileStorageType.remoteFileStore:
            uploadDatasetToServer(os.path.join(Constants.KGNET_Config.datasets_output_path, json_args["training"]["dataset_name"] + '.tsv'))
        elif Constants.KGNET_Config.fileStorageType == Constants.FileStorageType.S3:
            utils.uploadFileToS3(os.path.join(Constants.KGNET_Config.datasets_output_path, json_args["training"]["dataset_name"] + '.tsv'), file_type="metadata")
    return transform_results_dict, train_results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GMLaaS Pipeline')
    parser.add_argument('--path_json', type=str, default='args.json')
    parser.add_argument('--path_transformation_py', type=str, default='DataTransform/TSV_TO_PYG_dataset.py')
    parser.add_argument('--path_training_py', type=str, default='models/graph_saint_KGTOSA.py')
    args = parser.parse_args()
    cmd_run_training_pipeline(args.path_json, args.path_transformation_py, args.path_training_py)




