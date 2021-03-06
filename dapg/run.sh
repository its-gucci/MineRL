# debug model 
# python3 train_agent.py --pretrain 1 --bc 1 --episodes 1 --eval --eval-runs 1

# train model with no loading
# python3 train_agent.py

# load model and eval with no training
# python3 train_agent.py -l --eval 

# debug model on GCloud
python3 train_agent.py --pretrain 1 --bc 1 --episodes 1 --eval --eval-runs 1 --datadir '../distributed_rl/minerl_data/datasets'