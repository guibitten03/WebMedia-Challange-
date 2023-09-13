import time

from models.deepconn import DeepCoNN

from config.config import general_config


def choice_net(model):
    if model == "DeepCoNN":
        return DeepCoNN(general_config=general_config), general_config
    

def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))