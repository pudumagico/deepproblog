import json
import os
import pickle
from PIL import Image

from collections import Counter
from deepproblog.dataset import Dataset
from torch.utils.data import Dataset as TorchDataset
from problog.logic import Term, list2term, Constant
from deepproblog.query import Query

import utils
import torch

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import tqdm
from PIL import Image
from matplotlib.ticker import NullLocator
from torch.autograd import Variable
from torch.utils.data import DataLoader

from yolo.models import load_model
from yolo.utils.utils import PostprocessingMethod
from yolo.utils.datasets import ImageFolder
from yolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from yolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, to_cpu, print_environment_info, \
    non_max_suppression_enhanced

class ClevrTestDataset(Dataset, TorchDataset):
    def __init__(self, clevr_dir, img_size = 480, batch_size = 1, n_cpu=1, transform=None):

        quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_val_questions.json')
        self.img_dir = os.path.join(clevr_dir, 'images', 'val')
                
        with open(quest_json_filename, 'r') as json_file:
                self.questions = json.load(json_file)['questions']
                                
        self.clevr_dir = clevr_dir
        self.transform = transform
        dataset = ImageFolder(
            self.img_dir,
            transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))
        self.dataset = dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_cpu,
            pin_memory=True)
        self.dataloader = iter(dataloader)
            

    def __len__(self):
        return len(self.questions)
    
    def to_query(self, i):
        return Query(
            Term("ans", Term('X'))
        )

    def __getitem__(self, idx):
        current_question = self.questions[idx]
        image = next(self.dataloader)[1]
        question = current_question['question']
        answer = current_question['answer']
        program = current_question['program']
        
        sample = (image, answer, question , program)
        
        return sample