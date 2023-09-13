# -*- coding: utf-8 -*-

# Vou treinar com o conjunto de treino
# Validar com o conjunto de validação
# Testar com os itens cadidatos

# import fire
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import time
import pandas as pd
import fire


# FRAMEWORK IMPORTS
from Data.Data import NewsData
from Data.DataTest import TestData
from utils.utils import *
from framework.models import Model
from metrics.metrics import *
from config.config import *


# EXPERIMENT PARAMETERS 
seed = 2023
use_gpu = True
gpu_id = [0]
batch_size = 128


# MODEL PARAMETERS
lr = 2e-3
weight_decay = 1e-3


def train(model_name, num_epochs=10, path_loader="DataPreProcessing/data_pro"):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_gpu:
        torch.cuda.manual_seed_all(seed)
        
        if len(gpu_id) == 1:
            torch.cuda.set_device(gpu_id[0])

    net, config = choice_net(model_name)
    model = Model(config, net, model_name)

    model.to(device)
    if len(gpu_id) > 1:
        model = nn.DataParallel(model, device_ids=gpu_id)


    train = NewsData(
        path_loader, data_mode=train_test_config['data_mode'], user_doc_mode=train_test_config['user_doc_mode'])
    # train_data = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # val = NewsData(path_loader, data_mode='val')
    # val_data = DataLoader(val, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    print("start training....") 
    
    loss = nn.MSELoss()

    min_loss = 1e+10
    best_res = 1e+10

    for epoch in tqdm(range(num_epochs), desc="Training"):
        total_ndcg = 0.0
        total_loss = 0.0
        user_count = 0

        model.train()
        
        # Cada usuário será um batch.
        # O algoritmo irá calcular a similaridade entre o documento do usuário e entre o documento dos itens que ele tem na validação e dará um score de 0 a 1.
        # Depois, a lista será organizada com esse score.
        # Calcula o ndcg entre a lista organizada pelos scores do algoritmo e a lista verdadeira da validação.
        # Propaga o erro.

        
        for (user, itens) in tqdm(train.userIter.items(), desc="Users Counting"):

            
            # Primeiro pego o documento do usuário e dos itens
            user_doc, itens_doc = unpack_input(train, user, itens)
            
            # Depois crio uma lista de scores para prever o score para cada item
            # que o usuário consumiu no validação
            scores = []
            for i, item_doc in enumerate(itens_doc):
                item = itens[i]
                output = model([user, item, user_doc, item_doc])
                scores.append(output.item())
            
            
            # Com a lista de scores, consigo organizar a lista que o algoritmo previu
            # Calculo a relevancia original e a relevancia predita
            relevances = list(zip(itens, [(x+1) for x in range(len(itens))][::-1]))
            
            scores = list(zip(itens, scores))
            preds = sorted(scores, key=lambda x: x[1], reverse=True)
            preds = [(tuple[0], valor) for tuple, valor in zip(preds, range(len(preds), 0, -1))]
            
            preds = sorted(preds, key=lambda x: [t[0] for t in relevances].index(x[0]))

            relevances = torch.tensor([float(rel[1]) for rel in relevances], requires_grad=True)
            preds = torch.tensor([float(pred[1]) for pred in preds], requires_grad=True)
            
            # Consigo calcular a perda quadrática a partir dos scores de relevancia
            optimizer.zero_grad()
            rel_loss = loss(preds, relevances)
            total_loss += rel_loss.item()
            # print(loss)
            rel_loss.backward()
            optimizer.step()

            user_count += 1

            total_ndcg += ndcg_score(relevances, preds)

        
            # É calculado todos os erros de todos os usuários. O treinamento da rede ocorre ajustando
            # o erro de cada lista do usuário. Assim, a rede regula os pesos.
            # Calculem a média dos erros em cada época para observarem a convergencia do algoritmo

            # ndcg_score = ndcg_loss(relevances.detach().numpy(), preds.detach().numpy())

        total_loss = total_loss / user_count
        total_ndcg = total_ndcg / user_count
        print(f"Epoch [{epoch+1}/{num_epochs}], Test MSE: {total_loss}, Test NDCG: {total_ndcg}")
        
        if total_loss < min_loss:
            min_loss = total_loss
            model.save(name=model_name)



def test(model_name, path_loader="DataPreProcessing/data_pro/train"):

    model_path = f"checkpoints/{model_name}.pth"
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_gpu:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(gpu_id[0])

    net, config = choice_net(model_name)
    model = Model(config, net, model_name)

    model.cuda()

    model.load(model_path)

    data = TestData(path_loader, "datasets/out", user_doc_mode=train_test_config['user_doc_mode'])

    out = {
        "userId":[],
        "acessos_futuros":[]
    }
    

    # Preciso criar um loop que carrega um user e todos os itens da base com seus documentos
    for i, user in tqdm(enumerate(data.users), desc="Users Counting:"):
        user_doc, item_docs = unpack_input(data, user, data.itens[i])

        item_preds = []
        item_rank = []

        with torch.no_grad():
            for item, item_doc in zip(data.itens[i], item_docs):
                output = model([user, item, user_doc, item_doc])
                item_preds.append(output.item())
                item_rank.append(item)

        scores = list(zip(item_rank, item_preds))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        item_list_wise = [data.index2item[x[0]] for x in scores][:10]

        out['userId'] = out['userId'] + ([data.index2user[user]] * 10)
        out['acessos_futuros'] = out['acessos_futuros'] + (item_list_wise)


    
    out = pd.DataFrame(out)
    out.to_csv(f"results/out_{model_name}.csv", index=False)
    
        


def unpack_input(data_obj, user, itens):
    # Vetor do usuário já foi salvo como um tensor de shape(1, 768)
    user_doc = data_obj.userDoc[user].cuda()
    
    itens_doc = []
    for item in itens:
        itens_doc.append(
            torch.FloatTensor(data_obj.itemDoc[item]).unsqueeze(0).cuda()
            )
        
    
        
    return user_doc, itens_doc
    


if __name__ == "__main__":
    fire.Fire()
    # train("DeepCoNN")
    # test("DeepCoNN", "checkpoints/DeepCoNN.pth")