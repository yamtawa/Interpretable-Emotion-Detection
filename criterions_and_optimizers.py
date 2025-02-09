import torch.optim as optim
import torch.nn as nn
from load_config import *
config = load_config()




def CE_LOSS(output, true_labels):
    ce_loss = nn.CrossEntropyLoss()
    return ce_loss(output, true_labels)

def set_criterion(current_config):
    if current_config['CURRENT_STEP']['CRITERION_NAME']=='CE':
        return CE_LOSS


def set_optimizer(model,current_config):
    weight_decay = current_config['CURRENT_STEP'].get('WEIGHT_DECAY', 0)
    momentum = current_config['CURRENT_STEP'].get('MOMENTUM', 0)
    if current_config['CURRENT_STEP']['OPTIMIZER_NAME']=='Adam':
        return optim.Adam([{'params': model.bert.parameters(), 'lr': current_config['CURRENT_STEP']['LR_FEATURES']},
                {'params': model.classifier.parameters(), 'lr': current_config['CURRENT_STEP']['LR_HEAD']}],weight_decay=weight_decay)
    elif current_config['CURRENT_STEP']['OPTIMIZER_NAME']=='SGD':
        return optim.SGD([{'params': model.bert.parameters(), 'lr': current_config['CURRENT_STEP']['LR_FEATURES']},
                {'params': model.classifier.parameters(), 'lr': current_config['CURRENT_STEP']['LR_HEAD']}],  momentum=momentum, weight_decay=weight_decay)
    elif current_config['CURRENT_STEP']['OPTIMIZER_NAME'] == 'RMSprop':
        return optim.RMSprop([{'params': model.bert.parameters(), 'lr': current_config['CURRENT_STEP']['LR_FEATURES']},
                {'params': model.classifier.parameters(), 'lr': current_config['CURRENT_STEP']['LR_HEAD']}], weight_decay=weight_decay)