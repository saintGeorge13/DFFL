from collections import defaultdict
from dataclasses import replace
from email.policy import strict
import math
import random

import scipy

import utility
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as tu
from tqdm import tqdm
import wandb
import collections
import copy
class Trainer():
    def __init__(self, args, loader, agent_list, ckp):
        self.args = args
        self.ckp = ckp #save need modified
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.loaders_train = loader.loaders_train
        self.order = loader.order
        self.offset_worker = loader.offset_worker
        self.offset_sequence = loader.offset_sequence
        self.agent_list = agent_list[:-1] #last one is the tester
        self.tester = agent_list[-1]
        self.epoch = 0
        for agent in self.agent_list:
            agent.make_optimizer_all(ckp=ckp)
            agent.make_scheduler_all()
        self.tester.make_optimizer_all(ckp=ckp)
        self.tester.make_scheduler_all()
        
        self.run=wandb.init(project=args.project)
        self.run.name=args.save

        self.device = torch.device('cpu' if args.cpu else 'cuda')

        self.sync_at_init()

        if args.model.find('INQ') >= 0:
            self.inq_steps = args.inq_steps
        else:
            self.inq_steps = None

    def train(self, fairness_list):
        # epoch, _ = self.start_epoch()
        self.epoch += 1
        epoch = self.epoch
        
        for agent in self.agent_list:
            agent.make_optimizer_all()
            agent.make_scheduler_all(reschedule=epoch-1) #pls make sure if need resume  
        # Step 1: sample a list of agent w/o replacement
        if(self.args.fair):
            fair_list = []
            for i in range(self.args.n_agents):
                fair_list.append(self.agent_list[i].fair)
            
            b = np.asarray(fair_list).argsort()[-self.args.n_joined:]
            agent_joined = np.sort(b)
        else:
            agent_joined = np.sort(np.random.choice(range(self.args.n_agents),self.args.n_joined, replace=False))
        # Step 2: sample a list of associated budgets
        while True:
            if(self.args.skewed):
                agent_budget = []
                random_set = [0.0625] * 40 + [0.125]*15 + [0.25]*15 + [0.5]*15 + [1.0]*15
                for i in range(10):
                    agent_budget.append(random_set[random.randint(0, 99)])

                agent_budget =  np.asarray(agent_budget)
                np.random.shuffle(agent_budget)
                # agent_budget = [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.125, 0.25, 0.5, 1.0]
            else:
                agent_budget = np.random.choice(self.args.fraction_list, self.args.n_joined)
            # For implementation simiplicity, we sample all model size for all methods
            _, unique_counts = np.unique(agent_budget, return_counts=True)
            if len(unique_counts) == len(self.args.fraction_list):
                break
        # Step 3: create a buget -> client dictionary for syncing later
        budget_record = collections.defaultdict(list) #need to make sure is not empty
        for k, v in zip(agent_budget, agent_joined):
            budget_record[k].append(v)
        
        for i in agent_joined:
            self.agent_list[i].begin_all(epoch, self.ckp) #call move all to train() 
            self.agent_list[i].start_loss_log() #need check!

        self.start_epoch() #get current lr
        timer_model = utility.timer()
        #     timer_model.tic()

        # 1 HT-FL，10个用户，按照计算能力分序列，4个用户参与聚合
        # idx = 0
        # client_models = []
        # for net_f in self.args.fraction_list:
        #     agent_list_at_net_f = budget_record[net_f]
        #     for j in range(len(agent_list_at_net_f)):
        #         if j != 0:
        #             agent = self.agent_list[agent_list_at_net_f[j]]
        #             model_id = agent.budget_to_model(net_f)  
        #             agent.model_list[model_id].load_state_dict(copy.deepcopy(state_dict), strict=False)                    
        #         idx += 1
        #         timer_model.tic()
        #         loss, loss_orth, log_train = self.agent_list[agent_list_at_net_f[j]].train_local(self.loaders_train[agent_list_at_net_f[j]], net_f, self.args.local_epochs)
        #         timer_model.hold()
        #         tt=timer_model.release()
        #         self.ckp.write_log(
        #         '{}/{} ({:.0f}%)\t'
        #         'agent {}\t'
        #         'model {}\t'
        #         'NLL: {:.3f}\t'
        #         'Top1: {:.2f} / Top5: {:.2f}\t'
        #         'Total {:<2.4f}/ Orth: {:<2.5f} '
        #         'Time: {:.1f}s'.format(
        #             idx,
        #             len(agent_joined),
        #             100.0 * (idx) / len(agent_joined),idx,net_f,
        #             *(log_train),
        #             loss, loss_orth,
        #             tt
        #             )
        #         )

        #         model_id = self.tester.budget_to_model(net_f)
        #         state_dict = self.agent_list[agent_list_at_net_f[j]].model_list[model_id].state_dict()
            
        #     client_models.append(agent_list_at_net_f[j]) 
        # for i in agent_joined:
        #     self.agent_list[i].log_all(self.ckp)
        #     for loss in self.agent_list[i].loss_list:
        #         loss.end_log(len(self.loader_train.dataset)*self.args.local_epochs) #should be accurate
        # agent_joined2 = client_models
        # agent_budget2 = [0.25, 0.5, 0.75, 1]
        # budget_record2 = collections.defaultdict(list) #need to make sure is not empty
        # for k, v in zip(agent_budget2, agent_joined2):
        #     budget_record2[k].append(v)
        # self.sync(budget_record2, agent_joined2, agent_budget2)

        # 1+ HT-FL，10个用户，按照计算能力分序列,10个用户参与聚合
        # idx = 0
        # client_models = []
        # for net_f in self.args.fraction_list:
        #     agent_list_at_net_f = budget_record[net_f]
        #     for j in range(len(agent_list_at_net_f)):
        #         if j != 0:
        #             agent = self.agent_list[agent_list_at_net_f[j]]
        #             model_id = agent.budget_to_model(net_f)  
        #             agent.model_list[model_id].load_state_dict(copy.deepcopy(state_dict), strict=False)                    
        #         idx += 1
        #         timer_model.tic()
        #         loss, loss_orth, log_train = self.agent_list[agent_list_at_net_f[j]].train_local(self.loaders_train[agent_list_at_net_f[j]], net_f, self.args.local_epochs)
        #         timer_model.hold()
        #         tt=timer_model.release()
        #         self.ckp.write_log(
        #         '{}/{} ({:.0f}%)\t'
        #         'agent {}\t'
        #         'model {}\t'
        #         'NLL: {:.3f}\t'
        #         'Top1: {:.2f} / Top5: {:.2f}\t'
        #         'Total {:<2.4f}/ Orth: {:<2.5f} '
        #         'Time: {:.1f}s'.format(
        #             idx,
        #             len(agent_joined),
        #             100.0 * (idx) / len(agent_joined),idx,net_f,
        #             *(log_train),
        #             loss, loss_orth,
        #             tt
        #             )
        #         )

        #         model_id = self.tester.budget_to_model(net_f)
        #         state_dict = self.agent_list[agent_list_at_net_f[j]].model_list[model_id].state_dict()
            
        #     client_models.append(agent_list_at_net_f[j]) 
        # for i in agent_joined:
        #     self.agent_list[i].log_all(self.ckp)
        #     for loss in self.agent_list[i].loss_list:
        #         loss.end_log(len(self.loader_train.dataset)*self.args.local_epochs) #should be accurate
        
        # self.sync(budget_record, agent_joined, agent_budget)

        
    
        # 2 HT-FL，100个用户，按照数据分布分序列 useless
        # for n in range(0, self.args.hubs):           
        #     client_models = []
        #     for i in range(len(self.order[n])):    
        #         for j in range(len(self.order[n][i])):
        #             if j != 0:
        #                 agent = self.agent_list[self.order[n][i][j]+n*self.args.workers]
        #                 for net_f in self.args.fraction_list:
        #                     model_id = agent.budget_to_model(net_f)  
        #                     agent.model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)                    
        #             timer_model.tic()
        #             loss, loss_orth, log_train = self.agent_list[self.order[n][i][j]+n*self.args.workers].train_local(self.loaders_train[self.order[n][i][j]+n*self.args.workers], agent_budget[self.order[n][i][j]+n*self.args.workers], self.args.local_epochs)
        #             timer_model.hold()
        #             tt=timer_model.release()
        #             self.ckp.write_log(
        #             '{}/{} ({:.0f}%)\t'
        #             'agent {}\t'
        #             'model {}\t'
        #             'NLL: {:.3f}\t'
        #             'Top1: {:.2f} / Top5: {:.2f}\t'
        #             'Total {:<2.4f}/ Orth: {:<2.5f} '
        #             'Time: {:.1f}s'.format(
        #                 self.order[n][i][j]+n*self.args.workers,
        #                 len(agent_joined),
        #                 100.0 * (self.order[n][i][j]+n*self.args.workers) / len(agent_joined),self.order[n][i][j]+n*self.args.workers,agent_budget[self.order[n][i][j]+n*self.args.workers],
        #                 *(log_train),
        #                 loss, loss_orth,
        #                 tt
        #                 )
        #             )

                    
        #             filter_banks = {}
        #             for k,v in self.tester.model_list[0].state_dict().items():
        #                 if 'filter_bank' in k:
        #                     filter_banks[k] = torch.zeros(v.shape)
                    
        #             for k in filter_banks:
        #                 model_id = self.agent_list[self.order[n][i][j]+n*self.args.workers].budget_to_model(agent_budget[self.order[n][i][j]+n*self.args.workers])
        #                 state_dict = self.agent_list[self.order[n][i][j]+n*self.args.workers].model_list[model_id].state_dict()
        #                 filter_banks[k]+=state_dict[k]
                
        #         client_models.append(self.order[n][i][j]+n*self.args.workers) 
        #     filter_banks = {}
        #     for k,v in self.tester.model_list[0].state_dict().items():
        #         if 'filter_bank' in k:
        #             filter_banks[k] = torch.zeros(v.shape)
            
        #     for k in filter_banks:
        #         for i in range(len(client_models)):
        #             model_id = self.agent_list[client_models[i]].budget_to_model(agent_budget[client_models[i]])
        #             state_dict = self.agent_list[client_models[i]].model_list[model_id].state_dict()
        #             filter_banks[k]+=state_dict[k]*(self.offset_sequence[n][i] / np.sum(self.offset_sequence[n])) 
               
        #     for i in range(self.args.workers):
        #         for net_f in self.args.fraction_list:
        #             model_id = agent.budget_to_model(net_f)
        #             self.agent_list[i+n*self.args.workers].model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)
        # for i in agent_joined:
        #     self.agent_list[i].log_all(self.ckp)
        #     for loss in self.agent_list[i].loss_list:
        #         loss.end_log(len(self.loader_train.dataset)*self.args.local_epochs) #should be accurate
        # if epoch % self.args.q == 0:
        #     self.sync(budget_record, agent_joined, agent_budget)


        # 3 原始代码 useless
        # for idx, i in enumerate(agent_joined):
        #     timer_model.tic()

        #     loss, loss_orth, log_train = self.agent_list[i].train_local(self.loaders_train[i], 
        #                       agent_budget[idx], self.args.local_epochs)
            
        #     timer_model.hold()
        #     tt=timer_model.release()
            
        #     self.ckp.write_log(
        #             '{}/{} ({:.0f}%)\t'
        #             'agent {}\t'
        #             'model {}\t'
        #             'NLL: {:.3f}\t'
        #             'Top1: {:.2f} / Top5: {:.2f}\t'
        #             'Total {:<2.4f}/ Orth: {:<2.5f} '
        #             'Time: {:.1f}s'.format(
        #                 idx+1,
        #                 len(agent_joined),
        #                 100.0 * (idx+1) / len(agent_joined),i,agent_budget[idx],
        #                 *(log_train),
        #                 loss, loss_orth,
        #                 tt
        #                 )
        #             )
                  
        # for i in agent_joined:
        #     self.agent_list[i].log_all(self.ckp)
        #     for loss in self.agent_list[i].loss_list:
        #         loss.end_log(len(self.loader_train.dataset)*self.args.local_epochs) #should be accurate
        
        
        # self.sync(budget_record, agent_joined, agent_budget)

        # 3+ 原始代码, threepart + FedMLB
        for idx, i in enumerate(agent_joined):
            timer_model.tic()

            loss, log_train = self.agent_list[i].train_local2(self.loaders_train[i], 
                              agent_budget[idx], self.args.local_epochs)
            timer_model.hold()
            tt=timer_model.release()
            
            self.ckp.write_log(
                    '{}/{} ({:.0f}%)\t'
                    'agent {}\t'
                    'model {}\t'
                    'NLL: {:.3f}\t'
                    'Top1: {:.2f} / Top5: {:.2f}\t'
                    'Total {:<2.4f} '
                    'Time: {:.1f}s'.format(
                        idx+1,
                        len(agent_joined),
                        100.0 * (idx+1) / len(agent_joined),i,agent_budget[idx],
                        *(log_train),
                        loss, 
                        tt
                        )
                    )
            filter_banks = {}
            for k,v in self.tester.model_list[0].state_dict().items():
                if 'filter_bank' in k:
                    filter_banks[k] = torch.zeros(v.shape)
            
            for k in filter_banks:
                model_id = self.agent_list[i].budget_to_model(agent_budget[idx])
                state_dict = self.agent_list[i].model_list[model_id].state_dict()
                filter_banks[k]+=state_dict[k]    
            self.agent_list[i].filter = filter_banks  
            self.agent_list[i].select += 1
            

        for i in agent_joined:
            self.agent_list[i].log_all(self.ckp)
            for loss in self.agent_list[i].loss_list:
                loss.end_log(len(self.loader_train.dataset)*self.args.local_epochs) #should be accurate
        
        
        filter_banks_avg, anchors_avg = self.sync(budget_record, agent_joined, agent_budget)
        sum = 0
        for client in range(self.args.n_agents):
            total_norm = 0
            for p, P in zip(self.agent_list[client].filter.values(), filter_banks_avg.values()):
                param_norm = (p.data - P.data).norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            fair = self.offset_worker[client] / np.sum(np.asarray(self.offset_worker)) * total_norm / self.agent_list[client].select * 10
            self.agent_list[client].fair = fair
            sum += fair
        
        sum = sum / self.args.n_agents
        fairness_list.append(sum)

        # # 4 HT-FL，10个用户，按照数据分布分序列
        # distribution = []      
        # for i in agent_joined:
        #     if self.args.model == "resnet34":
        #         x = [0]*200
        #     else:
        #         x = [0]*10
        #     for e, data in enumerate(self.loaders_train[i]):
        #         image, label = data
        #         for j in label:
        #             x[int(j)] += 1
        #     distribution.append(x)
        # KL_div = []
        # if self.args.model == "resnet34":
        #     norm =  [0.005] * 200
        # else:
        #     norm =  [0.1] * 10
        
        # for i in range(len(agent_joined)):
        #     KL_div.append(round(scipy.stats.entropy(distribution[i], norm), 4))
        # KL_div = np.array(KL_div)
        # distribution = np.array(distribution)
        # order = []
        # kl = np.sort(KL_div)
        # kl_place = list(np.argsort(-KL_div))
        # tmp_list = [kl_place[0]]
        # tmp_distribution = distribution[kl_place[0]]
        # kl_place.remove(kl_place[0])
        # while len(kl_place) != 0:
        #     flag = False
        #     now_best_list = 0
        #     now_best_distribution = tmp_distribution
        #     for j in range(len(kl_place)):
        #         if scipy.stats.entropy(tmp_distribution + distribution[kl_place[j]], norm) < scipy.stats.entropy(now_best_distribution, norm):
        #             now_best_distribution = tmp_distribution + distribution[kl_place[j]]
        #             now_best_list = kl_place[j]
        #             flag = True
        #     if flag:     
        #         tmp_list.append(now_best_list)
        #         tmp_distribution = now_best_distribution
        #         kl_place.remove(now_best_list)
                
                
        #     if not flag or len(tmp_list) >= 3:
        #         order.append(tmp_list)
        #         if len(kl_place) != 0:
        #             tmp_list = [kl_place[0]]
        #             tmp_distribution = distribution[kl_place[0]]
        #             kl_place.remove(kl_place[0])
        #         else:
        #             tmp_list = []
        # if len(tmp_list) != 0:
        #     order.append(tmp_list)
                 
        # client_models = []
        # for i in range(len(order)):    
        #     for j in range(len(order[i])):
        #         if j != 0:
        #             agent = self.agent_list[agent_joined[order[i][j]]]
        #             for net_f in self.args.fraction_list:
        #                 model_id = agent.budget_to_model(net_f)  
        #                 agent.model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)                    
        #         timer_model.tic()
        #         loss, loss_orth, log_train = self.agent_list[agent_joined[order[i][j]]].train_local(self.loaders_train[agent_joined[order[i][j]]], agent_budget[order[i][j]], self.args.local_epochs)
        #         timer_model.hold()
        #         tt=timer_model.release()
        #         self.ckp.write_log(
        #         '{}/{} ({:.0f}%)\t'
        #         'agent {}\t'
        #         'model {}\t'
        #         'NLL: {:.3f}\t'
        #         'Top1: {:.2f} / Top5: {:.2f}\t'
        #         'Total {:<2.4f}/ Orth: {:<2.5f} '
        #         'Time: {:.1f}s'.format(
        #             order[i][j],
        #             len(agent_joined),
        #             100.0 * (order[i][j]) / len(agent_joined),agent_joined[order[i][j]],agent_budget[order[i][j]],
        #             *(log_train),
        #             loss, loss_orth,
        #             tt
        #             )
        #         )

                
        #         filter_banks = {}
        #         for k,v in self.tester.model_list[0].state_dict().items():
        #             if 'filter_bank' in k:
        #                 filter_banks[k] = torch.zeros(v.shape)
                
        #         for k in filter_banks:
        #             model_id = self.agent_list[agent_joined[order[i][j]]].budget_to_model(agent_budget[order[i][j]])
        #             state_dict = self.agent_list[agent_joined[order[i][j]]].model_list[model_id].state_dict()
        #             filter_banks[k]+=state_dict[k]
            
        #     client_models.append(order[i][j]) 
        # for i in agent_joined:
        #     self.agent_list[i].log_all(self.ckp)
        #     for loss in self.agent_list[i].loss_list:
        #         loss.end_log(len(self.loader_train.dataset)*self.args.local_epochs) #should be accurate
        
        # self.sync(budget_record, agent_joined, agent_budget)

    

        # 5 HT-FL，10个用户，按照数据分布分序列, +MLB
        # distribution = []      
        # for i in agent_joined:
        #     if self.args.model == "resnet34":
        #         x = [0]*200
        #     else:
        #         x = [0]*10
        #     for e, data in enumerate(self.loaders_train[i]):
        #         image, label = data
        #         for j in label:
        #             x[int(j)] += 1
        #     distribution.append(x)
        # KL_div = []
        # if self.args.model == "resnet34":
        #     norm =  [0.005] * 200
        # else:
        #     norm =  [0.1] * 10
        
        # for i in range(len(agent_joined)):
        #     KL_div.append(round(scipy.stats.entropy(distribution[i], norm), 4))
        # KL_div = np.array(KL_div)
        # distribution = np.array(distribution)
        # order = []
        # kl = np.sort(KL_div)
        # kl_place = list(np.argsort(-KL_div))
        # tmp_list = [kl_place[0]]
        # tmp_distribution = distribution[kl_place[0]]
        # kl_place.remove(kl_place[0])
        # while len(kl_place) != 0:
        #     flag = False
        #     now_best_list = 0
        #     now_best_distribution = tmp_distribution
        #     for j in range(len(kl_place)):
        #         if scipy.stats.entropy(tmp_distribution + distribution[kl_place[j]], norm) < scipy.stats.entropy(now_best_distribution, norm):
        #             now_best_distribution = tmp_distribution + distribution[kl_place[j]]
        #             now_best_list = kl_place[j]
        #             flag = True
        #     if flag:     
        #         tmp_list.append(now_best_list)
        #         tmp_distribution = now_best_distribution
        #         kl_place.remove(now_best_list)
                
                
        #     if not flag or len(tmp_list) >= 3:
        #         order.append(tmp_list)
        #         if len(kl_place) != 0:
        #             tmp_list = [kl_place[0]]
        #             tmp_distribution = distribution[kl_place[0]]
        #             kl_place.remove(kl_place[0])
        #         else:
        #             tmp_list = []
        # if len(tmp_list) != 0:
        #     order.append(tmp_list)
                 
        # client_models = []
        # for i in range(len(order)):    
        #     for j in range(len(order[i])):
        #         if j != 0:
        #             agent = self.agent_list[agent_joined[order[i][j]]]
        #             for net_f in self.args.fraction_list:
        #                 model_id = agent.budget_to_model(net_f)  
        #                 agent.model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)                    
        #         timer_model.tic()
        #         loss, log_train = self.agent_list[agent_joined[order[i][j]]].train_local2(self.loaders_train[agent_joined[order[i][j]]], agent_budget[order[i][j]], self.args.local_epochs)
        #         timer_model.hold()
        #         tt=timer_model.release()
        #         self.ckp.write_log(
        #         '{}/{} ({:.0f}%)\t'
        #         'agent {}\t'
        #         'model {}\t'
        #         'NLL: {:.3f}\t'
        #         'Top1: {:.2f} / Top5: {:.2f}\t'
        #         'Total {:<2.4f} '
        #         'Time: {:.1f}s'.format(
        #             order[i][j],
        #             len(agent_joined),
        #             100.0 * (order[i][j]) / len(agent_joined),agent_joined[order[i][j]],agent_budget[order[i][j]],
        #             *(log_train),
        #             loss,
        #             tt
        #             )
        #         )

                
        #         filter_banks = {}
        #         for k,v in self.tester.model_list[0].state_dict().items():
        #             if 'filter_bank' in k:
        #                 filter_banks[k] = torch.zeros(v.shape)
                
        #         for k in filter_banks:
        #             model_id = self.agent_list[agent_joined[order[i][j]]].budget_to_model(agent_budget[order[i][j]])
        #             state_dict = self.agent_list[agent_joined[order[i][j]]].model_list[model_id].state_dict()
        #             filter_banks[k]+=state_dict[k]
            
        #     client_models.append(order[i][j]) 
        
        # for i in agent_joined:
        #     self.agent_list[i].log_all(self.ckp)
        #     for loss in self.agent_list[i].loss_list:
        #         loss.end_log(len(self.loader_train.dataset)*self.args.local_epochs) #should be accurate
        
        # self.sync(budget_record, agent_joined, agent_budget)

        # 6 HT-FL，10个用户，按照数据分布分序列, + ScaleFL useless
        # distribution = []      
        # for i in agent_joined:
        #     if self.args.model == "resnet34":
        #         x = [0]*200
        #     else:
        #         x = [0]*10
        #     for e, data in enumerate(self.loaders_train[i]):
        #         image, label = data
        #         for j in label:
        #             x[int(j)] += 1
        #     distribution.append(x)
        # KL_div = []
        # if self.args.model == "resnet34":
        #     norm =  [0.005] * 200
        # else:
        #     norm =  [0.1] * 10
        
        # for i in range(len(agent_joined)):
        #     KL_div.append(round(scipy.stats.entropy(distribution[i], norm), 4))
        # KL_div = np.array(KL_div)
        # distribution = np.array(distribution)
        # order = []
        # kl = np.sort(KL_div)
        # kl_place = list(np.argsort(-KL_div))
        # tmp_list = [kl_place[0]]
        # tmp_distribution = distribution[kl_place[0]]
        # kl_place.remove(kl_place[0])
        # while len(kl_place) != 0:
        #     flag = False
        #     now_best_list = 0
        #     now_best_distribution = tmp_distribution
        #     for j in range(len(kl_place)):
        #         if scipy.stats.entropy(tmp_distribution + distribution[kl_place[j]], norm) < scipy.stats.entropy(now_best_distribution, norm):
        #             now_best_distribution = tmp_distribution + distribution[kl_place[j]]
        #             now_best_list = kl_place[j]
        #             flag = True
        #     if flag:     
        #         tmp_list.append(now_best_list)
        #         tmp_distribution = now_best_distribution
        #         kl_place.remove(now_best_list)
                
                
        #     if not flag or len(tmp_list) >= 3:
        #         order.append(tmp_list)
        #         if len(kl_place) != 0:
        #             tmp_list = [kl_place[0]]
        #             tmp_distribution = distribution[kl_place[0]]
        #             kl_place.remove(kl_place[0])
        #         else:
        #             tmp_list = []
        # if len(tmp_list) != 0:
        #     order.append(tmp_list)
                 
        # client_models = []
        # for i in range(len(order)):    
        #     for j in range(len(order[i])):
        #         if j != 0:
        #             agent = self.agent_list[agent_joined[order[i][j]]]
        #             for net_f in self.args.fraction_list:
        #                 model_id = agent.budget_to_model(net_f)  
        #                 agent.model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)                    
        #         timer_model.tic()
        #         loss, log_train = self.agent_list[agent_joined[order[i][j]]].train_local3(self.loaders_train[agent_joined[order[i][j]]], agent_budget[order[i][j]], self.args.local_epochs)
        #         timer_model.hold()
        #         tt=timer_model.release()
        #         self.ckp.write_log(
        #         '{}/{} ({:.0f}%)\t'
        #         'agent {}\t'
        #         'model {}\t'
        #         'NLL: {:.3f}\t'
        #         'Top1: {:.2f} / Top5: {:.2f}\t'
        #         'Total {:<2.4f} '
        #         'Time: {:.1f}s'.format(
        #             order[i][j],
        #             len(agent_joined),
        #             100.0 * (order[i][j]) / len(agent_joined),agent_joined[order[i][j]],agent_budget[order[i][j]],
        #             *(log_train),
        #             loss,
        #             tt
        #             )
        #         )

                
        #         filter_banks = {}
        #         for k,v in self.tester.model_list[0].state_dict().items():
        #             if 'filter_bank' in k:
        #                 filter_banks[k] = torch.zeros(v.shape)
                
        #         for k in filter_banks:
        #             model_id = self.agent_list[agent_joined[order[i][j]]].budget_to_model(agent_budget[order[i][j]])
        #             state_dict = self.agent_list[agent_joined[order[i][j]]].model_list[model_id].state_dict()
        #             filter_banks[k]+=state_dict[k]
            
        #     client_models.append(order[i][j]) 
        # # filter_banks = {}
        # # for k,v in self.tester.model_list[0].state_dict().items():
        # #     if 'filter_bank' in k:
        # #         filter_banks[k] = torch.zeros(v.shape)
        
        # # for k in filter_banks:
        # #     for i in range(len(client_models)):
        # #         model_id = self.agent_list[client_models[agent_joined[i]]].budget_to_model(agent_budget[client_models[i]])
        # #         state_dict = self.agent_list[client_models[agent_joined[i]]].model_list[model_id].state_dict()
        # #         filter_banks[k]+=state_dict[k]*(1 / len(client_models)) 
            
        # # for i in range(self.args.workers):
        # #     for net_f in self.args.fraction_list:
        # #         model_id = agent.budget_to_model(net_f)
        # #         self.agent_list[i+n*self.args.workers].model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)
        # for i in agent_joined:
        #     self.agent_list[i].log_all(self.ckp)
        #     for loss in self.agent_list[i].loss_list:
        #         loss.end_log(len(self.loader_train.dataset)*self.args.local_epochs) #should be accurate
        
        # self.sync(budget_record, agent_joined, agent_budget)



    def test(self):
        # epoch = self.scheduler.last_epoch + 1
        epoch = self.epoch
        self.ckp.write_log('\nEvaluation:')
        timer_test = utility.timer()
        self.tester.test_all(self.loader_test, timer_test, self.run, epoch)


    def sync_at_init(self): 
        if self.args.resume_from:
            for i in range(len(self.args.fraction_list)):
                print("resume from checkpoint")
                self.tester.model_list[i].load_state_dict(torch.load('../experiment/'+self.args.save+'/model/model_m'+str(i)+'_'+self.args.resume_from+'.pt'))
        # Sync all agents' parameters with tester before training 
        for net_f in self.args.fraction_list:
            model_id = self.tester.budget_to_model(net_f)
            state_dict = self.tester.model_list[model_id].state_dict()
            for agent in self.agent_list:
                agent.model_list[model_id].load_state_dict(copy.deepcopy(state_dict),strict=True)
            
    def sync(self, budget_record, agent_joined, agent_budget):
        # Step 1: gather all filter banks
        # This step runs across network fractions
        filter_banks = {}
        for k,v in self.tester.model_list[0].state_dict().items():
            if 'filter_bank' in k:
                filter_banks[k] = torch.zeros(v.shape)
        
        for k in filter_banks:
            for b, i in zip(agent_budget,agent_joined):
                model_id = self.agent_list[i].budget_to_model(b)
                state_dict = self.agent_list[i].model_list[model_id].state_dict()
                filter_banks[k]+=state_dict[k]*(self.offset_worker[i] / np.sum(np.array(self.offset_worker)[agent_joined]))
        # Step 2: gather all other parameters
        # This step runs within each network fraction
        anchors = {}
        for net_f in self.args.fraction_list:
            n_models = len(budget_record[net_f])
            agent_list_at_net_f = budget_record[net_f]
            
            anchor = {}
            model_id = self.tester.budget_to_model(net_f)
            for k, v in self.tester.model_list[model_id].state_dict().items():
                if 'filter_bank' not in k:
                    anchor[k] = torch.zeros(v.shape)
            for k in anchor:
                for i in agent_list_at_net_f:
                    model_id = self.agent_list[i].budget_to_model(net_f)
                    state_dict = self.agent_list[i].model_list[model_id].state_dict()
                    anchor[k]+=state_dict[k]*(1./n_models)
            anchors[net_f] = anchor 

        # Step 3: distribute anchors and filter banks to all agents
        for agent in self.agent_list:
            for net_f in self.args.fraction_list:
                model_id = agent.budget_to_model(net_f)
                agent.model_list[model_id].load_state_dict(copy.deepcopy(anchors[net_f]), strict=False)
                agent.model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)
        
        # Last step: update tester
        for net_f in self.args.fraction_list:
            model_id = self.tester.budget_to_model(net_f)
            self.tester.model_list[model_id].load_state_dict(copy.deepcopy(anchors[net_f]), strict=False)
            self.tester.model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)
        return filter_banks, anchors

    def prepare(self, *args):
        def _prepare(x):
            x = x.to(self.device)
            if self.args.precision == 'half': x = x.half()
            return x

        return [_prepare(a) for a in args]

    def start_epoch(self):

        lr = self.agent_list[0].scheduler_list[0].get_lr()[0]

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2}'.format(self.epoch, lr))

        return self.epoch, lr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            # epoch = self.scheduler.last_epoch + 1
            epoch = self.epoch
            return epoch >= self.args.epochs

    def _analysis(self):
        flops = torch.Tensor([
            getattr(m, 'flops', 0) for m in self.model.modules()
        ])
        flops_conv = torch.Tensor([
            getattr(m, 'flops', 0) for m in self.model.modules() if isinstance(m, nn.Conv2d)
        ])
        flops_ori = torch.Tensor([
            getattr(m, 'flops_original', 0) for m in self.model.modules()
        ])

        print('')
        print('FLOPs: {:.2f} x 10^8'.format(flops.sum() / 10**8))
        print('Compressed: {:.2f} x 10^8 / Others: {:.2f} x 10^8'.format(
            (flops.sum() - flops_conv.sum()) / 10**8 , flops_conv.sum() / 10**8
        ))
        print('Accel - Total original: {:.2f} x 10^8 ({:.2f}x)'.format(
            flops_ori.sum() / 10**8, flops_ori.sum() / flops.sum()
        ))
        print('Accel - 3x3 original: {:.2f} x 10^8 ({:.2f}x)'.format(
            (flops_ori.sum() - flops_conv.sum()) / 10**8,
            (flops_ori.sum() - flops_conv.sum()) / (flops.sum() - flops_conv.sum())
        ))
        input()

