import torch
# 在agent_federation 的 for batch, (img, label) in enumerate(loader_train): 卡住，请删掉对应数据集的dataLoader里的**kwargs
import utility
from data import Data
from model import Model
from loss import Loss
from trainer_FL import Trainer
from option import args
import random
import numpy as np
from model.agent_federation import Agent
import os

random.seed(0)
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
np.random.seed(0)
#print('Flag')
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.enabled=True
torch.backends.cudnn.deterministics=True

# This file create a list of agents...
# It also create a tester that sync at each communication for benchmarking
fairness_list = []
if checkpoint.ok:
    loader = Data(args)
    agent_list = [Agent(args, checkpoint, my_id) for my_id in range(args.n_agents)] #share ckp...need check if save
    tester = Agent(args, checkpoint, 1828) #a tester for runing test. Assign it a fixed id
    for agent in agent_list:
        agent.make_loss_all(Loss)
    tester.make_loss_all(Loss)
    #loss = Loss(args, checkpoint)
    t = Trainer(args, loader, agent_list+[tester], checkpoint)
    while not t.terminate():
        if agent_list[0].scheduler_list[0].last_epoch == -1 and not args.test_only:
            t.test()
        t.train(fairness_list)
        t.test()

    checkpoint.done()
print(fairness_list)
with open(os.path.join(os.path.join(args.dir_save, args.save), 'fair.txt'), 'w') as f:
    f.writelines(str(fairness_list))