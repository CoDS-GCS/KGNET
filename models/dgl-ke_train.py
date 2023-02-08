import re
import sys
import time
from dglke.train import main
from resource import *
if __name__ == '__main__':
    # sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    # sys.argv.pop(0)
    # print("sys.argv=",sys.argv)

#Default Paramters
# https://github.com/awslabs/dgl-ke/blob/master/examples/freebase/multi_cpu.sh

# dglke_train --model_name ComplEx --dataset Freebase --no_save_emb --log_interval 100 \
# --batch_size 1024 --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.1 --max_step 50000 \
# --batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --num_thread 1 --num_proc 48
    
#Large Graph
#     --model_name ComplEx --dataset FB15K --batch_size 1024 --neg_sample_size 256 --hidden_dim 50 \
# --gamma 143 --lr 0.015 --max_step 300000 --log_interval 10000 --batch_size_eval 1000  -adv \
# --no_save_emb --regularization_coef 2.00E-06  --num_thread 4 --num_proc 8 \
# --test --data_path /home/RGCN_LP/data/FB15K --format raw_udd_hrt --data_files train_profession_SQ.txt valid_profession.txt test_profession.txt --neg_sample_size_eval 10000 


    datasets=[
               'FB15K',
              # 'wn18',
              # 'Yago10',
              # 'wikikg-v2-2015'
              ]
    target_edges=[
            'profession',
            # '_hyponym',
            # 'isConnectedTo',
            #     'P57'
            ]
    subgraph=['SQ','BSQ','FG']
    idx=0
    for ds in datasets:
        print("######################## Dataset:"+ds+"#########################")
        for subg in subgraph:
            print("------subgraph:"+subg+"---------")
            start = time.time()
            sys.argv= ['dgl-ke_train.py','--model_name', 'ComplEx', 
                       '--dataset', ds, 
                       '--batch_size', '1024', 
                       '--neg_sample_size', '256', 
                       '--hidden_dim', '10', 
                       '--gamma', '143', 
                       '--lr', '0.1', 
                       '--max_step', '100', 
                       '--log_interval', '50', 
                       '--batch_size_eval', '1000',
                       '--neg_sample_size_eval','1000',
                       '-adv', 
                       # '--regularization_coef', '2.00E-06', 
                       '--num_thread', '8', 
                       '--num_proc', '8', 
                       '--no_save_emb',
                      '--test' ,
                       '--data_path' ,'/media/hussein/UbuntuData/GithubRepos/RGCN/data/'+ds,
                       '--format','raw_udd_hrt',
                       '--data_files','train'+
                       ('' if subg=='FG' else "_"+target_edges[idx]+"_"+subg)+".txt","valid_"+target_edges[idx]+
                       ".txt","test_"+target_edges[idx]+".txt",
                       '--neg_sample_size_eval' ,'10000' ]
            print("sys.argv=",sys.argv)
            print(getrusage(RUSAGE_SELF))
            main()
            print(getrusage(RUSAGE_SELF))
            print('subgraph total time takes {:.3f} seconds'.format(time.time() - start))
        idx+=1

