import scipy.stats
import numpy as np
def cluster(args, train_dataset, agent_dataid, loaders_train):
    offset_worker = []
    for n in range(0, args.hubs * args.workers):
        offset_worker.append(len(list(agent_dataid[n])))

    
    
    distribution = []
    for i in range(args.hubs):
        for j in range(args.workers):
            if args.model == "resnet34":
                x = [0]*200
            else:
                x = [0]*10
            for k in range(len(list(agent_dataid[args.workers*i+j]))):
                x[int(train_dataset[list(agent_dataid[args.workers*i+j])[k]][1])] += 1

            distribution.append(x)
    KL_div = []
    if args.model == "resnet34":
        norm =  [0.005] * 200
    else:
        norm =  [0.1] * 10
    
    for i in range(args.hubs*args.workers):
        KL_div.append(round(scipy.stats.entropy(distribution[i], norm), 4))
    KL_div = np.array(KL_div)
    print("KL of all workers: ", np.mean(np.array(KL_div)))
    distribution = np.array(distribution)
    order = []
    for i in range(args.hubs):
        kl = np.sort(KL_div[i*args.workers:(i+1)*args.workers])
        kl_place = list(np.argsort(-KL_div[i*args.workers:(i+1)*args.workers]))
        order_hub = []
        tmp_list = [kl_place[0]]
        tmp_distribution = distribution[kl_place[0] + args.workers*i]
        kl_place.remove(kl_place[0])
        while len(kl_place) != 0:
            flag = False
            now_best_list = 0
            now_best_distribution = tmp_distribution
            for j in range(len(kl_place)):
                if scipy.stats.entropy(tmp_distribution + distribution[kl_place[j] + args.workers*i], norm) < scipy.stats.entropy(now_best_distribution, norm):
                    now_best_distribution = tmp_distribution + distribution[kl_place[j] + args.workers*i]
                    now_best_list = kl_place[j]
                    flag = True
            if flag:     
                tmp_list.append(now_best_list)
                tmp_distribution = now_best_distribution
                kl_place.remove(now_best_list)
                    
                    
            if not flag or len(tmp_list) >= 3:
                order_hub.append(tmp_list)
                if len(kl_place) != 0:
                    tmp_list = [kl_place[0]]
                    tmp_distribution = distribution[kl_place[0] + args.workers*i]
                    kl_place.remove(kl_place[0])
                else:
                    tmp_list = []
        if len(tmp_list) != 0:
            order_hub.append(tmp_list)
        order.append(order_hub)
    offset_sequence = []
    for n in range(0, args.hubs):
        offset_sequence_temp = []
        for i in range(len(order[n])): 
            sum = 0
            for j in range(len(order[n][i])):
                sum += len(list(agent_dataid[order[n][i][j]+n*args.workers]))
            offset_sequence_temp.append(sum)
        offset_sequence.append(offset_sequence_temp)
    return offset_worker, offset_sequence, order