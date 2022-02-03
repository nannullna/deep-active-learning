import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=10000, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="MNIST", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="RandomSampling", 
                    choices=["RandomSampling", 
                             "LeastConfidence", 
                             "MarginSampling", 
                             "EntropySampling", 
                             "LeastConfidenceDropout", 
                             "MarginSamplingDropout", 
                             "EntropySamplingDropout", 
                             "KMeansSampling",
                             "KCenterGreedy", 
                             "BALDDropout", 
                             "AdversarialBIM", 
                             "AdversarialDeepFool"], help="query strategy")
args = parser.parse_args()
wandb.init(config=args, project="dal", name=args.strategy_name)
config = wandb.config
pprint(vars(config))
print()

# fix random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset = get_dataset(config.dataset_name)                   # load dataset
net = get_net(config.dataset_name, device)                   # load network
strategy = get_strategy(config.strategy_name)(dataset, net)  # load strategy

# start experiment
dataset.initialize_labels(config.n_init_labeled)
print(f"number of labeled pool: {config.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-config.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

# round 0 accuracy
print("Round 0")
strategy.train()
preds = strategy.predict(dataset.get_test_data())
test_acc = dataset.cal_test_acc(preds)
print(f"Round 0 testing accuracy: {test_acc}")
wandb.log({"test/accuracy": test_acc, "round": 0})

for rd in range(1, config.n_round+1):
    print(f"Round {rd}")

    # query
    query_idxs = strategy.query(config.n_query)

    # update labels
    strategy.update(query_idxs)
    strategy.train()

    # calculate accuracy
    preds = strategy.predict(dataset.get_test_data())
    test_acc = dataset.cal_test_acc(preds)
    print(f"Round {rd} testing accuracy: {test_acc}")
    wandb.log({"test/accuracy": test_acc, "round": rd})
