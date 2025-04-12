
import agent
import argparse



parser = argparse.ArgumentParser(description='Train or test model')
parser.add_argument('hyperparameters',help='')
parser.add_argument('--train',help='Training mode',action='store_true')
args = parser.parse_args()

dql = agent.Agent(hyperparameters_set=args.hyperparameters)
if args.train:
    dql.run(is_training=True,render=False)
else:
    dql.run(is_training=False,render=True)




