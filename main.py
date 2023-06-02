import os
import csv
import random
import tarfile
import multiprocessing as mp

import tqdm
import requests

import numpy as np
import sklearn.model_selection as skms

import torch
import torch.utils.data as td
import torch.nn.functional as F

import torchvision as tv
import torchvision.transforms.functional as TF

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# define constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = 'results'
RANDOM_SEED = 42

# create an output folder
os.makedirs(OUT_DIR, exist_ok=True)

def get_model_desc(pretrained=False, num_classes=200, use_attention=False):
    """
    Generates description string.
    """
    desc = list()

    if pretrained:
        desc.append('Transfer')
    else:
        desc.append('Baseline')

    if num_classes == 204:
        desc.append('Multitask')

    if use_attention:
        desc.append('Attention')

    return '-'.join(desc)


def log_accuracy(path_to_csv, desc, acc, sep='\t', newline='\n'):
    """
    Logs accuracy into a CSV-file.
    """
    file_exists = os.path.exists(path_to_csv)

    mode = 'a'
    if not file_exists:
        mode += '+'

    with open(path_to_csv, mode) as csv:
        if not file_exists:
            csv.write(f'setup{sep}accuracy{newline}')

        csv.write(f'{desc}{sep}{acc}{newline}')


import pandas as pd

with open('./data additional/image_attribute_labels.txt', 'r') as file:
    lines = file.readlines()

new_lines = []
for line in lines:
    line = line.strip().split(' ')
    if len(line) > 5:
        new_line = [x for x in line if x != '' and x != '0']
    else:
        new_line = line
    new_lines.append(new_line)

with open('./data additional/image_attribute_labels2.txt', 'w') as file:
    for line in new_lines:
        file.write(' '.join(line) + '\n')

account = pd.read_csv("./data additional/image_attribute_labels2.txt", sep=' ')
print("hola")

account.columns = ['id_imagen', 'id_atributo', 'valor si/no', 'certeza del valor', 'tiempo de anotacion']

# store dataframe into csv file
account.to_csv("./data additional/image_attribute_labels.csv", index=None)