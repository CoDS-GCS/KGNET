import argparse
import sys
import os
GMLaaS_models_path=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,GMLaaS_models_path)
# sys.path.append(sys.path.append(os.getcwd()+'/MorsE'))
# print('current dir is '+os.getcwd())
# print(sys.path)

from morseUtils import init_dir, set_seed, get_num_rel
from meta_trainer import MetaTrainer
from post_trainer import PostTrainer
import os.path as osp
from subgraph import gen_subgraph_datasets
from pre_process import data2pkl,data2pkl_Trans_to_Ind
from resource import *
import datetime
from morseUtils import Log
from Constants import *
import pandas as pd
import torch

def morse(dataset_name,root_path,modelID='',args=None,step='meta_train',seed=100,**kwargs):

    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    dict_results = {}
    if args is None:
        args = get_default_args()
        args.modelID = modelID


    args.ent_dim = args.emb_dim
    args.rel_dim = args.emb_dim
    args.name = args.dataset_name
    if args.kge in ['ComplEx', 'RotatE']:
        args.ent_dim = args.emb_dim * 2
    if args.kge in ['ComplEx']:
        args.rel_dim = args.emb_dim * 2

    def __init__():
        ''
    # print (f'creating dirs : {[args.state_dir,args.log_dir,args.tb_log_dir]}')
    # create_dir([args.state_dir,args.log_dir,args.tb_log_dir])
    logger = Log(args.log_dir, args.name).get_logger()
    init_dir(args)

    start_t = datetime.datetime.now()
    sample_start_t = datetime.datetime.now()
    args.data_path = root_path+dataset_name+"/"+dataset_name + '.pkl'
    args.db_path = root_path +  dataset_name + '_subgraph'
    # load original data and make index
    # print('checking data at ',args.data_path)
    if os.path.exists(args.data_path):
        os.remove(args.data_path)

    if args.loadTrainedModel == 0:
        if not os.path.exists(args.data_path):
            data2pkl_Trans_to_Ind(data_name=dataset_name, logger=logger, datapath=root_path)
            sample_time = str((datetime.datetime.now() - sample_start_t).total_seconds())
            logger.info("Sampling Time Sec=" + sample_time)
            dict_results['Sampling_Time'] = sample_time

            if not os.path.exists(args.db_path):
                print('generating subgraph')
                gen_subgraph_datasets(args)
            args.num_rel = get_num_rel(args)
            set_seed(seed)
            if step == 'meta_train':
                print('meta_train')
                start_train = datetime.datetime.now()
                meta_trainer = MetaTrainer(args, logger)
                model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
                # logger.info("BGP="+str(BGP))
                best_eval_rst = meta_trainer.train(datetime.datetime.now())
                print(f'best_eval_rst {best_eval_rst} type : {type(best_eval_rst)}')
                # end_t = datetime.datetime.now()
                train_time = str( (datetime.datetime.now() - start_train).total_seconds())
                total_time = str((datetime.datetime.now() - start_t).total_seconds())
                logger.info("Train Time Sec=" + train_time)
                logger.info("Total Time Sec=" + total_time)
            elif step == 'fine_tune':
                print('fine_tune')
                post_trainer = PostTrainer(args)
                post_trainer.train()
        model_trained_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
        dict_results["Final_Test_MRR"] = best_eval_rst['mrr']
        dict_results["Final_Test_Hits@10"] = best_eval_rst['hits@10']

        dict_results['Sampling_Time'] = sample_time
        dict_results['Train_Time'] = train_time
        dict_results['Total_Time'] = total_time
        dict_results['Results'] = dict(best_eval_rst)
        dict_results["init_ru_maxrss"] = init_ru_maxrss
        dict_results["model_ru_maxrss"] = model_loaded_ru_maxrss
        dict_results["model_trained_ru_maxrss"] = model_trained_ru_maxrss
        print('Done')
        return dict_results
    else:
        data2pkl_Trans_to_Ind(data_name=dataset_name, logger=logger, datapath=root_path)
        sample_time = str((datetime.datetime.now() - sample_start_t).total_seconds())
        logger.info("Sampling Time Sec=" + sample_time)
        dict_results['Sampling_Time'] = sample_time
        # gen_subgraph_datasets(args)
        args.num_rel = get_num_rel(args)
        set_seed(seed)
        meta_trainer = MetaTrainer(args, logger)
        meta_trainer.load_model(model_root=os.path.join(KGNET_Config.trained_model_path,'MorSE'),
                                model_name=args.modelID)


        ### DEBUGGING ###
        # kwargs['list_target_nodes'] = ["260881","260622","1332730","3122748","6066555"]
        # kwargs['target_rel'] = "_instance_hypernym"


        head_nodes = kwargs['list_target_nodes']
        target_rel = kwargs['target_rel']
        dataset_path = osp.join(root_path,dataset_name)
        entities_df = pd.read_csv(osp.join(dataset_path, 'entities.dict'), header=None, sep="\t")
        entities_dict = dict(zip(entities_df[1].astype(str), entities_df[0].astype(str)))
        rev_entities_dict = {str(v):str(k) for k,v in entities_dict.items()}

        relations_df = pd.read_csv(osp.join(dataset_path, 'relations.dict'), header=None, sep="\t")
        relations_dict = dict(zip(relations_df[1].astype(str), relations_df[0].astype(str)))

        encoded_target_nodes = [entities_dict[node] for node in head_nodes if node in entities_dict]
        encoded_relation = relations_dict[target_rel]

        inference_matrix = torch.zeros((len(encoded_target_nodes),3),dtype=torch.int32)
        inference_matrix[:,0] = torch.tensor([int(x) for x in encoded_target_nodes])
        inference_matrix[:,1] = torch.tensor(int(encoded_relation))

        results = meta_trainer.inference(triples=inference_matrix,k=2)
        y_pred = {}
        for node in head_nodes:
            if node not in entities_dict:
                y_pred[node] = 'None'
                continue
            index = encoded_target_nodes.index(entities_dict[node])
            y_pred[node] = [rev_entities_dict[str(pred)] for pred in results[index].tolist()]
        return y_pred


def create_dir(list_paths):
    for path in list_paths:
        if not os.path.exists(path):
            os.mkdir(path)


def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='wn18rr_inf')
    parser.add_argument('--name', default='_transe', type=str)
    parser.add_argument('--root_path', type=str, default=KGNET_Config.datasets_output_path)
    parser.add_argument('--loadTrainedModel', type=int, default=1)
    parser.add_argument('--modelID', type=str, default='wn18rr.model')

    parser.add_argument('--step', default='meta_train', type=str, choices=['meta_train', 'fine_tune'])
    # parser.add_argument('--metatrain_state', default='./state/wikikg-v2-2015/wikikg-v2-2015_transe.best', type=str)
    parser.add_argument('--metatrain_state', default=os.path.join(KGNET_Config.trained_model_path, 'MorSE') + '.best',
                        type=str)
    # parser.add_argument('--Target_rel',type=str,default='')

    # parser.add_argument('--state_dir', '-state_dir', default='./state', type=str)
    parser.add_argument('--state_dir', '-state_dir', default=os.path.join(KGNET_Config.trained_model_path, 'MorSE'),
                        type=str)
    parser.add_argument('--log_dir', '-log_dir', default='./log', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)

    # params for subgraph
    parser.add_argument('--num_train_subgraph', default=10000)
    parser.add_argument('--num_valid_subgraph', default=200)
    parser.add_argument('--num_sample_for_estimate_size', default=50)
    # parser.add_argument('--num_sample_for_estimate_size', default=150)
    parser.add_argument('--rw_0', default=10, type=int)
    parser.add_argument('--rw_1', default=10, type=int)
    parser.add_argument('--rw_2', default=5, type=int)
    parser.add_argument('--num_sample_cand', default=5, type=int)

    # params for meta-train
    parser.add_argument('--metatrain_num_neg', default=64)
    parser.add_argument('--metatrain_num_epoch', default=1)
    parser.add_argument('--metatrain_bs', default=64, type=int)
    parser.add_argument('--metatrain_lr', default=0.01, type=float)
    parser.add_argument('--metatrain_check_per_step', default=5, type=int)
    parser.add_argument('--indtest_eval_bs', default=512, type=int)

    # params for fine-tune
    parser.add_argument('--posttrain_num_neg', default=64, type=int)
    parser.add_argument('--posttrain_bs', default=512, type=int)
    parser.add_argument('--posttrain_lr', default=0.001, type=int)
    parser.add_argument('--posttrain_num_epoch', default=10, type=int)
    parser.add_argument('--posttrain_check_per_epoch', default=1, type=int)

    # params for R-GCN
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_bases', default=4, type=int)
    # parser.add_argument('--emb_dim', default=32, type=int)
    parser.add_argument('--emb_dim', default=32, type=int)

    # params for KGE
    parser.add_argument('--kge', default='TransE', type=str, choices=['TransE', 'DistMult', 'ComplEx', 'RotatE'])
    parser.add_argument('--gamma', default=10, type=float)
    parser.add_argument('--adv_temp', default=1, type=float)

    parser.add_argument('--gpu', default='cpu', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    args = parser.parse_args()
    create_dir([args.state_dir, args.log_dir, args.tb_log_dir])
    logger = Log(args.log_dir, args.name).get_logger()
    init_dir(args)
    return args

if __name__ == '__main__':
    args = get_default_args()
    morse(dataset_name=args.dataset_name,root_path=args.root_path,args=args,step=args.step,seed=args.seed)

    #
    # args.ent_dim = args.emb_dim
    # args.rel_dim = args.emb_dim
    # if args.kge in ['ComplEx', 'RotatE']:
    #     args.ent_dim = args.emb_dim * 2
    # if args.kge in ['ComplEx']:
    #     args.rel_dim = args.emb_dim * 2
    #
    # # specify the paths for original data and subgraph db
    #
    # Target_rel=args.Target_rel
    # #For BGP in ['BSQ','PQ','BPQ','FG']:
    # #for BGP in ['BSQ','FG','FG_ALL']:
    # start_t = datetime.datetime.now()
    # sample_start_t = datetime.datetime.now()
    # args.data_pathdata_pathdata_pathdata_path = args.root_path+'/'+args.data_name+'_.pkl'
    # args.db_path = args.root_path+'/'+args.data_name+'_subgraph'
    # # load original data and make index
    # if not os.path.exists(args.data_path):
    #     data2pkl_Trans_to_Ind(args.data_name,Target_rel,logger=logger, datapath=args.root_path)
    #     logger.info("Sampling Time Sec="+str((datetime.datetime.now() - sample_start_t).total_seconds()))
    #
    #     if not os.path.exists(args.db_path):
    #         gen_subgraph_datasets(args)
    #     args.num_rel = get_num_rel(args)
    #     set_seed(args.seed)
    #     if args.step == 'meta_train':
    #         meta_trainer = MetaTrainer(args,logger)
    #         # logger.info("BGP="+str(BGP))
    #         meta_trainer.train(datetime.datetime.now())
    #         end_t = datetime.datetime.now()
    #         logger.info("Total Time Sec="+str((datetime.datetime.now() - start_t).total_seconds()))
    #     elif args.step == 'fine_tune':
    #         post_trainer = PostTrainer(args)
    #         post_trainer.train()

        

