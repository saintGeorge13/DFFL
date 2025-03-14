with open("log.txt", "r", encoding='utf-8') as f:  #打开文本
    lines = f.readlines()   #读取文本
    i = 0
    list_acc = [[], [], [], []]
    list_loss = [[], [], [], []]
    list_best_acc = [[], [], [], []]
    for line in lines[1:]:

        if i == 16:
            list_acc[0].append(float(line[18:]))
        elif i == 20:
            list_acc[1].append(float(line[18:]))
        elif i == 24:
            list_acc[2].append(float(line[18:]))
        elif i == 28:
            list_acc[3].append(float(line[18:]))
        
        
        if i == 13:
            list_loss[0].append(float(line[14: 19]))
        if i == 17:
            list_loss[1].append(float(line[14: 19]))
        if i == 21:
            list_loss[2].append(float(line[14: 19]))
        if i == 25:
            list_loss[3].append(float(line[14: 19]))
        


        if i == 14:
            list_best_acc[0].append(round(100 - float(line[34: 40]), 2))
        if i == 18:
            list_best_acc[1].append(round(100 - float(line[34: 40]), 2))
        if i == 22:
            list_best_acc[2].append(round(100 - float(line[34: 40]), 2))
        if i == 26:
            list_best_acc[3].append(round(100 - float(line[34: 40]), 2))
        

        i = (i + 1) % 29



with open("acc.txt", "w") as f:   
    for i in range(4):
        f.writelines(str(list_acc[i]) + "\n")

with open("loss.txt", "w") as f:   
    for i in range(4):
        f.writelines(str(list_loss[i]) + "\n")

with open("best_acc.txt", "w") as f:   
    for i in range(4):
        f.writelines(str(list_best_acc[i]) + "\n")
       
    