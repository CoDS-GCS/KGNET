import argparse
# import sys
# sys.path.append('..')
from MorsE.main import morse
from MorsE.utils import init_dir,Log
import Constants

def run_morse(dataset_name,root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='FB15K23')
    parser.add_argument('--name', default='FB15K23', type=str)
    parser.add_argument('--root_path', type=str, default=Constants.KGNET_Config.datasets_output_path)

    parser.add_argument('--step', default='meta_train', type=str, choices=['meta_train', 'fine_tune'])
    parser.add_argument('--metatrain_state', default=Constants.KGNET_Config.datasets_output_path+'./state/WikiKG2_LP/WikiKG2_LP_transe.best', type=str)
    # parser.add_argument('--Target_rel',type=str,default='isConnectedTo')

    parser.add_argument('--state_dir', '-state_dir', default=Constants.KGNET_Config.datasets_output_path+'trained_models/', type=str)
    parser.add_argument('--log_dir', '-log_dir', default=Constants.KGNET_Config.datasets_output_path+'logs/', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)

    # params for subgraph
    parser.add_argument('--num_train_subgraph', default=100)
    parser.add_argument('--num_valid_subgraph', default=50)
    parser.add_argument('--num_sample_for_estimate_size', default=50)
    #parser.add_argument('--num_sample_for_estimate_size', default=150)
    parser.add_argument('--rw_0', default=10, type=int)
    parser.add_argument('--rw_1', default=10, type=int)
    parser.add_argument('--rw_2', default=5, type=int)
    parser.add_argument('--num_sample_cand', default=5, type=int)

    # params for meta-train
    parser.add_argument('--metatrain_num_neg', default=32)
    parser.add_argument('--metatrain_num_epoch', default=5)
    parser.add_argument('--metatrain_bs', default=32, type=int)
    parser.add_argument('--metatrain_lr', default=0.01, type=float)
    parser.add_argument('--metatrain_check_per_step', default=4, type=int)
    parser.add_argument('--indtest_eval_bs', default=512, type=int)

    # params for fine-tune
    parser.add_argument('--posttrain_num_neg', default=32, type=int)
    parser.add_argument('--posttrain_bs', default=64, type=int)
    parser.add_argument('--posttrain_lr', default=0.001, type=int)
    parser.add_argument('--posttrain_num_epoch', default=2, type=int)
    parser.add_argument('--posttrain_check_per_epoch', default=1, type=int)

    # params for R-GCN
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_bases', default=4, type=int)
    #parser.add_argument('--emb_dim', default=32, type=int)
    parser.add_argument('--emb_dim', default=128, type=int)

     # params for KGE
    parser.add_argument('--kge', default='TransE', type=str, choices=['TransE', 'DistMult', 'ComplEx', 'RotatE'])
    parser.add_argument('--gamma', default=10, type=float)
    parser.add_argument('--adv_temp', default=1, type=float)

    parser.add_argument('--gpu', default='cpu', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('-f') ## dummy args
    args = parser.parse_args()
    args.dataset_name = dataset_name
    args.root_path = root_path
    args.name = args.dataset_name
    return morse(dataset_name=args.dataset_name, root_path=args.root_path, args=args,
          step=args.step, seed=args.seed)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='FB15K23')
    parser.add_argument('--name', default='FB15K23', type=str)
    parser.add_argument('--root_path', type=str, default=Constants.KGNET_Config.datasets_output_path)

    parser.add_argument('--step', default='meta_train', type=str, choices=['meta_train', 'fine_tune'])
    parser.add_argument('--metatrain_state', default=Constants.KGNET_Config.datasets_output_path+'./state/WikiKG2_LP/WikiKG2_LP_transe.best', type=str)
    # parser.add_argument('--Target_rel',type=str,default='isConnectedTo')

    parser.add_argument('--state_dir', '-state_dir', default=Constants.KGNET_Config.datasets_output_path+'trained_models/', type=str)
    parser.add_argument('--log_dir', '-log_dir', default=Constants.KGNET_Config.datasets_output_path+'logs/', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)

    # params for subgraph
    parser.add_argument('--num_train_subgraph', default=100)
    parser.add_argument('--num_valid_subgraph', default=50)
    parser.add_argument('--num_sample_for_estimate_size', default=50)
    #parser.add_argument('--num_sample_for_estimate_size', default=150)
    parser.add_argument('--rw_0', default=10, type=int)
    parser.add_argument('--rw_1', default=10, type=int)
    parser.add_argument('--rw_2', default=5, type=int)
    parser.add_argument('--num_sample_cand', default=5, type=int)

    # params for meta-train
    parser.add_argument('--metatrain_num_neg', default=16)
    parser.add_argument('--metatrain_num_epoch', default=2)
    parser.add_argument('--metatrain_bs', default=32, type=int)
    parser.add_argument('--metatrain_lr', default=0.01, type=float)
    parser.add_argument('--metatrain_check_per_step', default=4, type=int)
    parser.add_argument('--indtest_eval_bs', default=512, type=int)

    # params for fine-tune
    parser.add_argument('--posttrain_num_neg', default=16, type=int)
    parser.add_argument('--posttrain_bs', default=64, type=int)
    parser.add_argument('--posttrain_lr', default=0.001, type=int)
    parser.add_argument('--posttrain_num_epoch', default=2, type=int)
    parser.add_argument('--posttrain_check_per_epoch', default=1, type=int)

    # params for R-GCN
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_bases', default=4, type=int)
    #parser.add_argument('--emb_dim', default=32, type=int)
    parser.add_argument('--emb_dim', default=32, type=int)

     # params for KGE
    parser.add_argument('--kge', default='TransE', type=str, choices=['TransE', 'DistMult', 'ComplEx', 'RotatE'])
    parser.add_argument('--gamma', default=10, type=float)
    parser.add_argument('--adv_temp', default=1, type=float)

    parser.add_argument('--gpu', default='cpu', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    args = parser.parse_args()
    args.name = args.dataset_name
    logger = Log(args.log_dir, args.name).get_logger()
    init_dir(args)

    best_eval_rst = morse(dataset_name=args.dataset_name, root_path=args.root_path, args=args,
          step=args.step, seed=args.seed)
    print(best_eval_rst)




