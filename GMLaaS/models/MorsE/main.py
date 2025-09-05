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
from pre_process import data2pkl,data2pkl_Trans_to_Ind,data2pkl_FG_inf,data2pkl_WISE_inf
from resource import *
import datetime
from morseUtils import Log
from Constants import *
import pandas as pd
import torch

def morse(dataset_name,root_path=KGNET_Config.datasets_output_path,modelID='',args=None,step='meta_train',seed=100,inf_dataset_name=None,inf_root_path = None,**kwargs):

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
    # if os.path.exists(args.data_path):
    #     os.remove(args.data_path)

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
        # data2pkl_Trans_to_Ind(data_name=dataset_name, logger=logger, datapath=root_path)
        # data2pkl_FG_inf (data_name=dataset_name, datapath=root_path)
        if inf_dataset_name is not None:
            data2pkl_WISE_inf(data_name=inf_dataset_name, datapath=inf_root_path)
        else:
            # data2pkl_Trans_to_Ind(data_name=dataset_name, logger=logger, datapath=root_path)
            data2pkl_FG_inf (data_name=dataset_name, datapath=root_path)

        sample_time = str((datetime.datetime.now() - sample_start_t).total_seconds())
        logger.info("Sampling Time Sec=" + sample_time)
        dict_results['Sampling_Time'] = sample_time
        # gen_subgraph_datasets(args)
        args.num_rel = get_num_rel(args)

        set_seed(seed)
        meta_trainer = MetaTrainer(args, logger)
        meta_trainer.load_model(model_root=os.path.join(KGNET_Config.trained_model_path,'MorSE'),
                                model_name=args.modelID)
        if inf_dataset_name is not None:
            args.data_path = inf_root_path+inf_dataset_name + '/'+inf_dataset_name+'.pkl'
        # meta_trainer.evaluate_valid_subgraphs()

        # data2pkl_FG_inf(data_name=dataset_name, logger=logger, datapath=root_path)
        ### DEBUGGING ###
#         kwargs['list_target_nodes'] = ["http://www.yago3-10/airport/Kansai_International_Airport","http://www.yago3-10/airport/Dubai_International_Terminal_3",
# 'http://www.yago3-10/airport/Skopje_"Alexander_the_Great"_Airport']
#         kwargs['target_rel'] = 'http://www.yago3-10/isConnectedTo'



        dataset_path = osp.join(root_path,dataset_name)
        import pickle
        # with open(os.path.join(dataset_path,dataset_name+'.pkl'),'rb') as f:
        #     data = pickle.load(f)
        #     entities = data['ent_reidx']
        #     relations = data['fix_rel_reidx']
        #     del data

        # entities_df = pd.read_csv(osp.join(dataset_path, 'entities.dict'), header=None, sep="\t")
        # entities_dict = dict(zip(entities_df[1].astype(str), entities_df[0].astype(str)))
        # rev_entities_dict = {str(v):str(k) for k,v in entities_dict.items()}

        # relations_df = pd.read_csv(osp.join(dataset_path, 'relations.dict'), header=None, sep="\t")
        # relations_dict = dict(zip(relations_df[1].astype(str), relations_df[0].astype(str)))

        target_triples = pd.read_csv(osp.join(dataset_path,'test.txt'),header=None,sep='\t',names=['s','p','o'])
        target_triples = target_triples.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        head_nodes = target_triples['s'].to_list()[:args.NUM_NODES]
        target_rel = target_triples['p'][0]
        tail_label = target_triples['o'].to_list()[:args.NUM_NODES]

        out = meta_trainer.inf_FG(args=args)
        print(out)
        import sys
        sys.exit()
        # head_nodes = kwargs['list_target_nodes']
        # target_rel = kwargs['target_rel']
        encoded_target_nodes = [int(entities[node]) for node in head_nodes if node in entities]
        encoded_relation = int(relations[target_rel])
        """ uncomment the line below! """
        y_true =  [int(entities[node]) for node in tail_label if node in tail_label]

        inference_matrix = torch.zeros((len(encoded_target_nodes),3),dtype=torch.int32)
        inference_matrix[:,0] = torch.tensor([int(x) for x in encoded_target_nodes])
        inference_matrix[:,1] = torch.tensor(int(encoded_relation))

        out = meta_trainer.inf_FG(args=args)
        out = meta_trainer.inference(triples=inference_matrix,k=10)
        import pickle
        with open ('/home/afandi/GitRepos/KGNET/morse_preds.pkl','rb') as f:
            out = pickle.load(f)
        pred = out
        y_true = torch.tensor(y_true)
        y_true_bool = torch.zeros((len(y_true), pred.shape[1]), dtype=torch.bool)
        y_true_bool[torch.arange(len(y_true)), y_true] = True
        b_range = torch.arange(pred.size()[0],)
        target_pred = pred[b_range, y_true]
        pred = torch.where(y_true_bool, -torch.ones_like(pred) * 10000000, pred)
        pred[b_range, y_true] = target_pred

        tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                       dim=1, descending=False)[b_range, y_true]
        ranks = tail_ranks.float()
        count = 0
        count += torch.numel(ranks)
        from collections import defaultdict as ddict
        results = ddict(float)#{'mr':0,'mrr':0}
        results['mr'] += torch.sum(ranks).item()
        results['mrr'] += torch.sum(1.0 / ranks).item()

        for k in [1, 5, 10]:
            results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        for k, v in results.items():
            results[k] = v / count
        print(results)

        # results are topK indices

        # To get True Label Names
        # y_pred = {}
        # for node in head_nodes:
        #     if node not in entities_dict:
        #         y_pred[node] = 'None'
        #         continue
        #     index = encoded_target_nodes.index(entities_dict[node])
        #     y_pred[node] = [rev_entities_dict[str(pred)] for pred in results[index].tolist()]

        true_pred_count = 0
        for target in range(results.shape[0]):
            if y_true[target] in results[target].tolist():
                true_pred_count += 1

        hits_10 = true_pred_count / args.NUM_NODES
        print(hits_10)

        return #y_pred


def create_dir(list_paths):
    for path in list_paths:
        if not os.path.exists(path):
            os.mkdir(path)


def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='YAGO_3-10_isConnectedTo_FG')# DBLP2023-010305  YAGO_3-10_isConnectedTo_FG WikiKG2_LP_P106_FG
    parser.add_argument('--name', default='_transe', type=str)
    parser.add_argument('--root_path', type=str, default=KGNET_Config.datasets_output_path)
    parser.add_argument('--loadTrainedModel', type=int, default=1)
    parser.add_argument('--modelID', type=str, default='YAGO_3-10_isConnectedTo_FG.model')# DBLP2023-010305 #YAGO_3-10_isConnectedTo_FG #WikiKG2_LP_P106_FG
    parser.add_argument('--NUM_NODES', type=int, default=100)


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

        

