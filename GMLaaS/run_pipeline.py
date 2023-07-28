import json
import pandas as pd
import subprocess
import argparse



def load_args(path_json):
    with open (path_json) as json_file:
        json_args = json.load(json_file)
    return json_args
        
def format_args(task,json_args,path_script):
    format_args = []
    format_args.extend(['python',path_script])
    for arg in json_args[task]:
        string_arg = '--'+str(arg)+'='+str(json_args[task][arg])
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
# list_training.extend(['python','Training/models/graph_saint/graph_saint_KGTOSA.py'])
# # list_training.append('Training/models/graph_saint/graph_saint_KGTOSA.py')
# for arg in data['training_args'].keys():
#     format_arg = '--'+str(arg)+'='+str(data['training_args'][arg])
#     list_training.append(format_arg)

# subprocess.run(list_training)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GMLaaS Pipeline')
    parser.add_argument('--path_json',type=str,default='args.json')
    parser.add_argument('--path_transformation_py',type=str,default='DataTransform/TSV_TO_PYG_dataset.py')
    parser.add_argument('--path_training_py',type=str,default='Training/models/graph_saint/graph_saint_KGTOSA.py')
    args = parser.parse_args()
    
    json_args = load_args(args.path_json)
    args_transformation = format_args(task='transformation',json_args=json_args,path_script=args.path_transformation_py)
    args_training = format_args (task = 'training',json_args=json_args,path_script=args.path_training_py)
    subprocess.run(args_transformation)
    subprocess.run(args_training)


