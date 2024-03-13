# from copy import copy
# import json
# import argparse
import os

import shutil
from tqdm import tqdm
import datetime
from Constants import *
from GMLaaS.DataTransform.TSV_TO_PYG_dataset import transform_tsv_to_PYG #inference_transform_tsv_to_PYG
import pandas as pd
from resource import *
import faulthandler
faulthandler.enable()
import pickle
import warnings
import zarr
from concurrent.futures import ProcessPoolExecutor, as_completed,ThreadPoolExecutor
from SparqlMLaasService.TaskSampler.TOSG_Extraction_NC import get_d1h1_TargetListquery
import zipfile


def execute_query_v2(batch, inference_file,kg, graph_uri='http://wikikg-v2',):
    # batch = ['<' + target + '>' if '<' not in target or '>' not in target else target for target in batch]

    # query = get_d1h1_TargetListquery(graph_uri=graph_uri,target_lst=batch)
    subgraph_df = kg.KG_sparqlEndpoint.executeSparqlquery(batch)#kg.KG_sparqlEndpoint.execute_sparql_multithreads([query], inference_file)
    if len(subgraph_df.columns) != 3:
        print(subgraph_df)
        raise AssertionError
    subgraph_df = subgraph_df.applymap(lambda x: x.strip('"'))
    if os.path.exists(inference_file):
        subgraph_df.to_csv(inference_file, header=None, index=None, sep='\t', mode='a')
    else:
        subgraph_df.to_csv(inference_file, index=None, sep='\t', mode='a')
def batch_tosa_v2 (targetNodesList,inference_file,graph_uri,kg,BATCH_SIZE=2000):
    def batch_generator():
        ptr = 0
        while ptr < len(targetNodesList):
            yield targetNodesList[ptr:ptr + BATCH_SIZE]
            ptr += BATCH_SIZE

    queries = []
    for q in batch_generator():
        target_lst = ['<' + target + '>' for target in q]
        queries.extend(get_d1h1_TargetListquery(graph_uri=graph_uri, target_lst=target_lst))
    if len(targetNodesList) > BATCH_SIZE:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(execute_query_v2, batch, inference_file, kg, graph_uri) for batch in
                       queries]
            for future in tqdm(futures, desc="Downloading Raw Graph", unit="subgraphs"):
                # future.result()
                pass
    else:
        [execute_query_v2(batch,inference_file,kg,graph_uri) for batch in queries]
def execute_query_v0(batch, inference_file, graph_uri,kg):

    # formatted_links = batch
    batch = ['<' + target + '>' if '<' not in target or '>' not in target else target for target in batch]

    query = get_d1h1_TargetListquery(graph_uri=graph_uri,target_lst=batch)
    subgraph_df = kg.KG_sparqlEndpoint.executeSparqlquery(query)
    if len(subgraph_df.columns) != 3:
        print(subgraph_df)
        raise AssertionError
    subgraph_df = subgraph_df.applymap(lambda x: x.strip('"'))
    if os.path.exists(inference_file):
        subgraph_df.to_csv(inference_file, header=None, index=None, sep='\t', mode='a')
    else:
        subgraph_df.to_csv(inference_file, index=None, sep='\t', mode='a')

# def batch_tosa(path_target_csv,inference_file,graph_uri,BATCH_SIZE=2000):
#     def batch_generator():
#         ptr = 0
#         while ptr < len(df_targets):
#             yield df_targets.iloc[ptr:ptr + BATCH_SIZE, :]
#             ptr += BATCH_SIZE
#
#     df_targets = pd.read_csv(path_target_csv, header=None)
#
#     """ Parallel Threading"""
#     if len(df_targets) > BATCH_SIZE:
#         with ProcessPoolExecutor() as executor:
#             futures = [executor.submit(execute_query, batch, inference_file, graph_uri) for batch in batch_generator()]
#
#             # Wait for all tasks to complete
#             for future in tqdm(futures, desc="Downloading Raw Graph", unit="subgraphs"):
#                 # future.result()
#                 pass
#
#     else:
#         execute_query(df_targets,inference_file,kg)

def execute_query(batch, inference_file, kg):
    formatted_links = ','.join(['<' + link + '>' for link in batch[0]])
    query = """
    PREFIX dblp2022: <https://dblp.org/rdf/schema#> 
    PREFIX kgnet: <http://kgnet/> 
    SELECT DISTINCT ?s ?p ?o
    FROM <http://dblp.org>  
    WHERE
     {{
        ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> dblp2022:Publication .
        ?s dblp2022:publishedIn ?dblp_Venue . 
        ?s dblp2022:title ?Title . 
        ?s ?p ?o.
        FILTER(?s IN ({formatted_links}))
     }}
    """

    query = """
        PREFIX dblp2022: <https://dblp.org/rdf/schema#>
        PREFIX kgnet: <http://kgnet/>

        SELECT DISTINCT ?s ?p ?o
        from <http://dblp.org>
        where
        {{
        ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> dblp2022:Publication .
        ?s dblp2022:publishedIn ?dblp_Venue .
        ?s dblp2022:title ?Title .
        #?s <https://dblp.org/rdf/schema#yearOfPublication> "2022".
        ?s <https://dblp.org/rdf/schema#publishedIn> ?conf .
        filter(?conf in ("AAAI","ACC","Appl. Math. Comput.","Autom.","BMC Bioinform.","Bioinform.","CDC","CVPR","CoRR","Commun. ACM","Discret. Math.","EMBC","EUSIPCO","Eur. J. Oper. Res.","Expert Syst. Appl.","GLOBECOM","HICSS","IACR Cryptol. ePrint Arch.","ICASSP","ICC","ICIP","ICRA","IECON","IEEE Access","IEEE Trans. Autom. Control.","IEEE Trans. Commun.","IEEE Trans. Geosci. Remote. Sens.","IEEE Trans. Ind. Electron.","IEEE Trans. Inf. Theory","IEEE Trans. Signal Process.","IEEE Trans. Veh. Technol.","IGARSS","IJCAI","IJCNN","INTERSPEECH","IROS","ISCAS","ISIT","Inf. Sci.","Lecture Notes in Computer Science","Multim. Tools Appl.","NeuroImage","Neurocomputing","PIMRC","Remote. Sens.","SMC","Sensors","Theor. Comput. Sci.","WCNC","WSC")) .
        ?s ?p ?o.
        FILTER(?s IN ({formatted_links}))
        }}

    """
    # query = """
    #     select distinct (?s as ?subject) (?p as ?predicate) (?o as ?object)
    #     from <http://wikikg-v2>
    #     where
    #     {
    #     ?s ?p ?o.
    #     }
    #     limit ?limit
    #     offset ?offset
    #  """
    # query = query.format(formatted_links=formatted_links)
    # subgraph_df = kg.KG_sparqlEndpoint.executeSparqlquery(query)
    subgraph_df = kg.KG_sparqlEndpoint.execute_sparql_multithreads([query], inference_file)
    if len(subgraph_df.columns) != 3:
        print(subgraph_df)
        raise AssertionError
    subgraph_df = subgraph_df.applymap(lambda x: x.strip('"'))
    if os.path.exists(inference_file):
        subgraph_df.to_csv(inference_file, header=None, index=None, sep='\t', mode='a')
    else:
        subgraph_df.to_csv(inference_file, index=None, sep='\t', mode='a')

""" Functions for Loading Mappings in Generate_inference_subgraph ()"""

def process_node(node,master_mapping,inference_mapping):
    try:
        df_master = pd.read_csv(os.path.join(master_mapping, node), dtype=str)
        df_inf = pd.read_csv(os.path.join(inference_mapping, node), dtype=str)

        intersection = pd.merge(df_master, df_inf, on='ent name', how='inner', suffixes=('_orig', '_inf'))

        return node.split('_entidx2name')[0], intersection
    except Exception as e:
        if node == 'relidx2relname.csv.gz' or node == 'labelidx2labelname.csv.gz':
            key = node.split('.csv.gz')[0]
            col_name = 'rel name' if node.startswith('rel') else 'label name'
            intersection = pd.merge(df_master, df_inf, on=col_name, how='inner', suffixes=('_orig', '_inf'))

            return key, intersection
        else:
            raise Exception(f"Unhandled Node {node}: {e}")

def process_all_nodes(nodes,master_mapping,inference_mapping):
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_node, node,master_mapping,inference_mapping): node for node in nodes}
        results = {}

        for future in as_completed(futures):
            node = futures[future]
            try:
                result = future.result()
                results[result[0]] = result[1]
            except Exception as e:
                print(f'Error processing node {node}: {str(e)}')

    return results
def getLabelMapping(dataset_name):
    path_labels = os.path.join(KGNET_Config.datasets_output_path,dataset_name,'mapping','labelidx2labelname.csv.gz')
    df_labels = pd.read_csv(path_labels)
    return df_labels.set_index('label idx')['label name'].to_dict()

def fill_missing_rel (relation,dir):
    # warnings.warn('@@@@@@@ FILLING MISSING TRIPLE {} @@@@@@@'.format(relation), UserWarning)
    # dir = os.path.join(inference_relations, relation)
    os.mkdir(dir)
    edge_data = {0: ["-1"], 1: ["-1"]}
    pd.DataFrame(edge_data).to_csv(os.path.join(dir, 'edge.csv.gz'), header=None, index=None,
                                   compression='gzip')  # edge.csv
    pd.DataFrame(edge_data).to_csv(os.path.join(dir, 'edge_reltype.csv.gz'), header=None, index=None,
                                   compression='gzip')  # edge_reltype.csv
    pd.DataFrame({0: ["1"]}).to_csv(os.path.join(dir, 'num-edge-list.csv.gz'), header=None, index=None,
                                    compression='gzip')


def store_emb(model,model_name,root_path=os.path.join(KGNET_Config.trained_model_path,'emb_store'),chunk_size=128):
    def zip_directory(directory_to_zip, output_zip_file):
        with zipfile.ZipFile(output_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            total_dirs = sum(len(dirs) for _, dirs, _ in os.walk(directory_to_zip))
            with tqdm(total=total_dirs, desc='Zipping directories') as pbar:
                for root, dirs, _ in os.walk(directory_to_zip):
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        arcname = os.path.relpath(dir_path, os.path.dirname(directory_to_zip))
                        zipf.write(dir_path, arcname=arcname)
                        pbar.update(1)  
                        for _, _, files in os.walk(dir_path):
                            for file in files:
                                file_path = os.path.join(dir_path, file)
                                arcname = os.path.relpath(file_path, os.path.dirname(directory_to_zip))
                                zipf.write(file_path, arcname=arcname)

    path = os.path.join(root_path,model_name)
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    emb_store = zarr.DirectoryStore(path)
    root = zarr.group(store=emb_store)

    emb_mapping = {}
    # Iterate over node types in emb_dict
    ptr = 0
    for key, val in model.emb_dict.items():
        # Convert PyTorch tensor to a NumPy array with a compatible data type
        val_np = val.cpu().detach().numpy()

        # Create a Zarr array for each node type
        emb_array = root.create(key, shape=val_np.shape, dtype=val_np.dtype, chunks=(chunk_size, -1)) #chunks=(val_np.shape[0], -1)
        emb_array[:] = val_np  # Assign the embeddings to the Zarr array

        # Update the mapping information
        # emb_mapping[key] = {'start': 0, 'end': val_np.shape[0]}
        emb_mapping[key] = (ptr,ptr+val_np.shape[0]-1)
        ptr+=val_np.shape[0]#+1
    # Save the mapping information
    emb_mapping_path = os.path.join(path, 'index.map')
    with open(emb_mapping_path, 'wb') as f:
        pickle.dump(emb_mapping, f)

    zip_directory(path,path+'.zip')


def generate_inference_subgraph(master_ds_name, graph_uri='',targetNodesList = [],labelNode = None,targetNodeType=None,
                                target_rel_uri='https://dblp.org/rdf/schema#publishedIn',
                                ds_types = '',
                                sparqlEndpointURL=KGNET_Config.KGMeta_endpoint_url,
                                output_file='inference_subG'):


    global download_end_time
    time_ALL_start = datetime.datetime.now()

    inference_file = os.path.join(KGNET_Config.inference_path, output_file + '.tsv')

    download_start_time = datetime.datetime.now()
    if os.path.exists(inference_file):
        os.remove(inference_file)
    from KGNET import KGNET
    kg = KGNET(sparqlEndpointURL, KG_Prefix='http://kgnet/') #TODO: Parameterize
    # batch_tosa(path_target_csv=r'/home/afandi/GitRepos/KGNET/Datasets/TARGET_1000d2.csv', #TODO parameterize
    #            inference_file=inference_file,
    #            kg=kg)

    batch_tosa_v2(targetNodesList,inference_file=inference_file,graph_uri=graph_uri,kg=kg)
    download_end_time = (datetime.datetime.now() - download_start_time).total_seconds()
    # print(f"******** DOWNLOAD_TIME : {download_end_time}")

    """ Non Batched Query execution """
    # subgraph_df = kg.KG_sparqlEndpoint.executeSparqlquery(inference_node_query)
    # subgraph_df = subgraph_df.applymap(lambda x : x.strip('"'))
    # subgraph_df.to_csv(inference_file,index=None,sep='\t')

    """ Mapping node ids !"""
    # master_ds_root = r'/home/afandi/GitRepos/KGNET/Datasets/mid-0000114_orig' #TODO: Replace with arg based directory
    master_ds_root = os.path.join(KGNET_Config.datasets_output_path, master_ds_name)
    master_mapping = os.path.join(master_ds_root, 'mapping')
    master_relations = os.path.join(master_ds_root, 'raw', 'relations')

    inference_root = os.path.join(KGNET_Config.inference_path, output_file)
    inference_mapping = os.path.join(inference_root, 'mapping')
    inference_relations = os.path.join(inference_root, 'raw', 'relations')

    if os.path.exists(inference_root):
        shutil.rmtree(inference_root)

    if os.path.exists(inference_root + '.zip'):
        os.remove(inference_root + '.zip')

    if not os.path.exists(master_ds_root):
        if os.path.exists(master_ds_root+'.zip'):
            shutil.unpack_archive(master_ds_root+'.zip', KGNET_Config.datasets_output_path)
        else:
            raise Exception('No Master Graph at {} '.format(master_ds_root))

    time_infTransform_start = datetime.datetime.now()
    transform_tsv_to_PYG(dataset_name=output_file,
                                  dataset_name_csv=output_file,
                                  #dataset_types=os.path.join(KGNET_Config.datasets_output_path,'WikiKG2015_v2_Types.csv'), #TODO: Replace with arg based file # For DBLP /home/afandi/GitRepos/KGNET/Datasets/dblp2022_Types (rec).csv
                                  dataset_types=os.path.join(KGNET_Config.datasets_output_path, ds_types+"_Types.csv"),
                                  target_rel =target_rel_uri,#,publishedIn
                                  targetNodeType=targetNodeType, #TODO: Parameterize
                                  output_root_path=KGNET_Config.inference_path,
                                  Header_row=0,
                                  labelNodetype = labelNode,
                                  split_rel=None,   ### For Transform_tsv_to_PYG
                                  similar_target_rels=[],### For Transform_tsv_to_PYG
                                  inference=True### For Transform_tsv_to_PYG
                                   )
    time_infTransform_end = (datetime.datetime.now() - time_infTransform_start).total_seconds()
    target_node = \
    [x for x in os.listdir(os.path.join(inference_root, 'split')) if not (x.endswith('.csv') | x.endswith('.gz'))][0]
    time_mapLoad_start = datetime.datetime.now()
    inf_edges = os.listdir(inference_relations)

    """ Loading Mapping Parallel"""
    #
    # Use glob for file listing
    import glob
    master_nid = glob.glob(os.path.join(master_mapping, '*.csv.gz'))
    inf_nid = glob.glob(os.path.join(inference_mapping, '*.csv.gz'))
    common_nodes = set(os.path.basename(node) for node in inf_nid).intersection(
        os.path.basename(node) for node in master_nid)
    mapping_dict = process_all_nodes(common_nodes, master_mapping, inference_mapping)
    """ ********************** """
    # global target_masks  # To be used for filtering inference nodes from training nodes during inference
    # target_masks = [int(x) for x in mapping_dict[target_node]['ent idx_orig'].tolist()]
    # time_mapLoad_end = (datetime.datetime.now() - time_mapLoad_start).total_seconds()
    if not len(targetNodesList) == len(mapping_dict[target_node]): # If the extracted subgraph contains more target nodes than the one in inference
        df = pd.merge(pd.DataFrame({'ent name':targetNodesList}),mapping_dict[target_node],on='ent name',how='left')
        # target_masks = [int(x) for x in df['ent idx_orig'].tolist()]
        target_masks = [int(x) for x in mapping_dict[target_node]['ent idx_orig'].tolist()]
        target_masks_inf = [int(x) for x in df['ent idx_inf'].tolist()]
        del df
    else:
        target_masks = [int (x) for x in mapping_dict[target_node]['ent idx_orig'].tolist()]

    time_map_start = datetime.datetime.now()
    num_success_relations = 0
    num_failed_relations = 0
    num_missing_relations = 0
    num_master_relations = len(master_relations)
    for triple in inf_edges:
        try:
            src, rel, dst = triple.split('___')
            src_df = mapping_dict[src]
            dst_df = mapping_dict[dst]
            directory = os.path.join(inference_relations, triple)
            edge_df = pd.read_csv(os.path.join(directory, 'edge.csv.gz'), header=None, dtype=str, compression='gzip')
            src_merged_df = pd.merge(edge_df, src_df, left_on=0, right_on='ent idx_inf', how='left')  # how='inner'
            dst_merged_df = pd.merge(edge_df, dst_df, left_on=1, right_on='ent idx_inf', how='left')  # how='inner'
            edge_df[0] = src_merged_df['ent idx_orig']
            edge_df[1] = dst_merged_df['ent idx_orig']

            edge_rel_df = pd.read_csv(os.path.join(directory, 'edge_reltype.csv.gz'), header=None, dtype=str,
                                      compression='gzip')

            if edge_df.isnull().values.any():
                warnings.warn('______ DROPPING UNKNOWN ROWS IN {} ______'.format(triple), UserWarning)
                nan_rows_index = edge_df[edge_df.isnull().any(axis=1)].index
                edge_df = edge_df.dropna()
                edge_rel_df = edge_rel_df.drop(index=nan_rows_index)
                """ Modify the number of rows """
                pd.DataFrame({0: [str(len(edge_df))]}).to_csv(os.path.join(directory, 'num-edge-list.csv.gz'),
                                                              header=None, index=None,
                                                              compression='gzip')  # num-edge-list
            if len(edge_df) == 0:
                # edge_df = pd.DataFrame({0:["-1"],1:["-1"]}).to_csv(os.path.join(directory,'edge.csv.gz'),header=None,index=None,compression='gzip')
                edge_df = pd.DataFrame({0: ["-1"], 1: ["-1"]})
                edge_df.to_csv(os.path.join(directory, 'edge.csv.gz'), header=None, index=None,
                               compression='gzip')  # num-edge-list
                edge_df.to_csv(os.path.join(directory, 'edge_reltype.csv.gz'), header=None, index=None,
                               compression='gzip')
                pd.DataFrame({0: ["1"]}).to_csv(os.path.join(directory, 'num-edge-list.csv.gz'), header=None,
                                                index=None, compression='gzip')
                num_failed_relations += 1
            else:
                edge_df.to_csv(os.path.join(directory, 'edge.csv.gz'), header=None, index=None, compression='gzip')
                rel_df = mapping_dict['relidx2relname']
                current_rel_id = edge_rel_df[0][0]
                edge_rel_df[0] = rel_df[rel_df['rel idx_inf'] == str(current_rel_id)]['rel idx_orig'].iloc[0]
                edge_rel_df.to_csv(os.path.join(directory, 'edge_reltype.csv.gz'), header=None, index=None,
                                   compression='gzip')
                num_success_relations += 1

        except Exception as e:
            print(e)
            warnings.warn('****** SKIPPING UNKNOWN TRIPLE {} ******'.format(triple), UserWarning)
    """ Map labels"""
    inf_label_dir = os.path.join(inference_root,'raw','node-label')
    for label in os.listdir(inf_label_dir):
            label_csv = os.path.join(inf_label_dir,label,'node-label.csv.gz')
            label_df = pd.read_csv(label_csv,header=None,dtype=str,compression='gzip')
            mapping_dict['labelidx2labelname']['label idx_inf'] = mapping_dict['labelidx2labelname']['label idx_inf'].astype(int)
            intersection = pd.merge(label_df.astype(float).astype(int), mapping_dict['labelidx2labelname'], left_on=0, right_on='label idx_inf',
                                    how='left').fillna(-1)
            missing_values = sum(intersection['label idx_orig']==-1)
            if missing_values > 0:
                warnings.warn('............. {}  MISSING LABELS out of {}, .i.e {:.2f}% .............'.format(missing_values,len(label_df),(missing_values/len(label_df))*100), UserWarning)
            intersection['label idx_orig'].to_csv(label_csv,header=None,index=None,compression='gzip')

    time_map_end = (datetime.datetime.now() - time_map_start).total_seconds()
    """ Filling missing relations"""
    time_fill_start = datetime.datetime.now()
    missing_relations = set(os.listdir(master_relations)) - set(os.listdir(inference_relations))
    unknown_relations = set(os.listdir(inference_relations)) - set(os.listdir(master_relations))
    print(f'Unknown relations: {unknown_relations}')
    with ProcessPoolExecutor() as executor:
        # items = [(relation,os.path.join(inference_relations,relation)) for relation in missing_relations]
        futures = [executor.submit(fill_missing_rel,relation,os.path.join(inference_relations,relation)) for relation in missing_relations]
        for future in tqdm(futures,desc='Inserting Missing Relations', unit='relation'):
            pass
    num_missing_relations = len(missing_relations)
        # for file in os.listdir(dir):
        #     with open(os.path.join(dir,file),'rb') as f_in, gzip.open(os.path.join(dir,file+'.gz'),'wb') as f_out:
        #         f_out.writelines(f_in)
    time_fill_end = (datetime.datetime.now() - time_fill_start).total_seconds()


    """ Move meta-data inside 'raw' folder """
    time_move_start = datetime.datetime.now()
    master_meta_files = [file for file in os.listdir(os.path.join(master_ds_root, 'raw')) if (
        file.endswith('.gz'))]  # or file.endswith('.gz')) ]# and not file.startswith('num-node-dict')]
    for file in master_meta_files:
        dst_file = os.path.join(inference_root, 'raw', file)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        shutil.copyfile(os.path.join(master_ds_root, 'raw', file), dst_file)

    """ Zip the final file """
    processed_dir = os.path.join(KGNET_Config.inference_path, output_file)
    if os.path.exists(processed_dir + '.zip'):
        os.remove(processed_dir + '.zip')

    tmp_dir = os.path.join(KGNET_Config.inference_path, 'tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    os.makedirs(tmp_dir)
    tmp_parent = os.path.join(tmp_dir, output_file)
    os.mkdir(tmp_parent)
    shutil.copytree(processed_dir, tmp_parent, dirs_exist_ok=True)
    shutil.make_archive(processed_dir, 'zip', tmp_dir)

    shutil.rmtree(tmp_dir)
    time_move_end = (datetime.datetime.now() - time_move_start).total_seconds()
    time_ALL_end = (datetime.datetime.now() - time_ALL_start).total_seconds()
    print(8 * "*", ' DOWNLOAD TIME ', download_end_time, "*" * 8)
    print(8 * "*", ' TRANSFORMATION TIME ', time_infTransform_end, "*" * 8)
    # print(8 * "*", ' LOAD MAPPINGS TIME ', time_mapLoad_end, "*" * 8)
    print(8 * "*", ' MAPPING TIME ', time_map_end, "*" * 8)
    print(8 * "*", ' FILLING TIME ', time_fill_end, "*" * 8)
    print(8 * "*", ' MOV AND ZIP TIME ', time_move_end, "*" * 8)
    print(8 * "*", ' TOTAL TIME', time_ALL_end, "*" * 8)
    print(8 * "*", ' TOTAL Master Relations ', num_master_relations, "*" * 8)
    print(8 * "*", ' SUCCESS RELATIONS ', num_success_relations, "*" * 8)
    print(8 * "*", ' FAILED RELATIONS ', num_failed_relations, "*" * 8)
    print(8 * "*", ' MISSING RELATIONS ', num_missing_relations, "*" * 8)
    print(8 * "*", ' UNKNOWN RELATIONS at INF ', len(unknown_relations), "*" * 8)
    print('*' * 8, ' Max RAM Usage: ', getrusage(RUSAGE_SELF).ru_maxrss / (1024 * 1024), ' GB')

    # sys.exit()
    if "target_masks_inf" in locals():
        return output_file,target_masks,target_masks_inf  # .replace('.zip','')
    else:
        return output_file,target_masks,None