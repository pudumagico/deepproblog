import json
import re
import sys
import pickle
sys.path.append('../../')

from dataGen import termPath2dataList
from network import Net
# from neurasp import NeurASP
from translate import func_to_asp, json_to_facts, parse_facts
from tqdm import tqdm
import utils
import sys
from json import dumps

import torch
from torchvision import transforms
from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
# from ....deepproblog.evaluate import get_confusion_matrix
from yolo.detect import create_data_loader
# from yolo.models import Darknet
from yolo.models import load_model
from yolo.utils.utils import load_classes, non_max_suppression


from deepproblog.heuristics import geometric_mean
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.utils import get_configuration, format_time_precise, config_to_string
from dataset import ClevrTestDataset

neural_atom = r'''
    nn(label, [1], (I,B) , [obj(B,cylinder,large,gray,metal,X1,Y1,X2,Y2), obj(B,sphere,large,gray,metal,X1,Y1,X2,Y2), obj(B,cube,large,gray,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,gray,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,gray,rubber,X1,Y1,X2,Y2), obj(B,cube,large,gray,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,blue,metal,X1,Y1,X2,Y2), obj(B,sphere,large,blue,metal,X1,Y1,X2,Y2), obj(B,cube,large,blue,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,blue,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,blue,rubber,X1,Y1,X2,Y2), obj(B,cube,large,blue,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,brown,metal,X1,Y1,X2,Y2), obj(B,sphere,large,brown,metal,X1,Y1,X2,Y2), obj(B,cube,large,brown,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,brown,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,brown,rubber,X1,Y1,X2,Y2), obj(B,cube,large,brown,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,yellow,metal,X1,Y1,X2,Y2), obj(B,sphere,large,yellow,metal,X1,Y1,X2,Y2), obj(B,cube,large,yellow,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,yellow,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,yellow,rubber,X1,Y1,X2,Y2), obj(B,cube,large,yellow,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,red,metal,X1,Y1,X2,Y2), obj(B,sphere,large,red,metal,X1,Y1,X2,Y2), obj(B,cube,large,red,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,red,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,red,rubber,X1,Y1,X2,Y2), obj(B,cube,large,red,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,green,metal,X1,Y1,X2,Y2), obj(B,sphere,large,green,metal,X1,Y1,X2,Y2), obj(B,cube,large,green,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,green,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,green,rubber,X1,Y1,X2,Y2), obj(B,cube,large,green,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,purple,metal,X1,Y1,X2,Y2), obj(B,sphere,large,purple,metal,X1,Y1,X2,Y2), obj(B,cube,large,purple,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,purple,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,purple,rubber,X1,Y1,X2,Y2), obj(B,cube,large,purple,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,cyan,metal,X1,Y1,X2,Y2), obj(B,sphere,large,cyan,metal,X1,Y1,X2,Y2), obj(B,cube,large,cyan,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,cyan,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,cyan,rubber,X1,Y1,X2,Y2), obj(B,cube,large,cyan,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,gray,metal,X1,Y1,X2,Y2), obj(B,sphere,small,gray,metal,X1,Y1,X2,Y2), obj(B,cube,small,gray,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,gray,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,gray,rubber,X1,Y1,X2,Y2), obj(B,cube,small,gray,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,blue,metal,X1,Y1,X2,Y2), obj(B,sphere,small,blue,metal,X1,Y1,X2,Y2), obj(B,cube,small,blue,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,blue,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,blue,rubber,X1,Y1,X2,Y2), obj(B,cube,small,blue,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,brown,metal,X1,Y1,X2,Y2), obj(B,sphere,small,brown,metal,X1,Y1,X2,Y2), obj(B,cube,small,brown,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,brown,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,brown,rubber,X1,Y1,X2,Y2), obj(B,cube,small,brown,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,yellow,metal,X1,Y1,X2,Y2), obj(B,sphere,small,yellow,metal,X1,Y1,X2,Y2), obj(B,cube,small,yellow,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,yellow,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,yellow,rubber,X1,Y1,X2,Y2), obj(B,cube,small,yellow,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,red,metal,X1,Y1,X2,Y2), obj(B,sphere,small,red,metal,X1,Y1,X2,Y2), obj(B,cube,small,red,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,red,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,red,rubber,X1,Y1,X2,Y2), obj(B,cube,small,red,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,green,metal,X1,Y1,X2,Y2), obj(B,sphere,small,green,metal,X1,Y1,X2,Y2), obj(B,cube,small,green,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,green,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,green,rubber,X1,Y1,X2,Y2), obj(B,cube,small,green,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,purple,metal,X1,Y1,X2,Y2), obj(B,sphere,small,purple,metal,X1,Y1,X2,Y2), obj(B,cube,small,purple,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,purple,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,purple,rubber,X1,Y1,X2,Y2), obj(B,cube,small,purple,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,cyan,metal,X1,Y1,X2,Y2), obj(B,sphere,small,cyan,metal,X1,Y1,X2,Y2), obj(B,cube,small,cyan,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,cyan,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,cyan,rubber,X1,Y1,X2,Y2), obj(B,cube,small,cyan,rubber,X1,Y1,X2,Y2)]) :: box(I,B,X1,Y1,X2,Y2).
    '''

correct = 0
incorrect = 0
invalid = 0
total = 0

clevr_dataset_test = ClevrTestDataset('./CLEVR_v1.0')

with open("problog_theory.lp", "r") as fp:
    theory = fp.read()

program = theory
program += '\n'
program += neural_atom

config_path = './yolo/yolov3_scene_parser.cfg'
weights_path = './yolo/yolov3_ckpt_200.pth'
yolo = load_model(config_path, weights_path)

net = Network(yolo, "label", batching=False)

model = Model(program, [net], load=False)
model.set_engine(ExactEngine(model), cache=True)

for i, data in enumerate(clevr_dataset_test):

    question = func_to_asp(data[3])
    query = clevr_dataset_test.to_query(i)
    model.add_question(question)
    answer = model.solve([query])[0]
    print(answer)
    actual = data[1]

#     if len(answer.result) == 0:
#         predicted = "no_answer"
#         # if verbose > 1:
#         #     print("no answer for query {}".format(gt_query))
#     else:
#         max_ans = max(answer.result, key=lambda x: answer.result[x])
#         p = answer.result[max_ans]
#         if eps is None:
#             predicted = str(max_ans.args[gt_query.output_ind[0]])
#         else:
#             predicted = float(max_ans.args[gt_query.output_ind[0]])
#             actual = float(gt_query.output_values()[0])
#             if abs(actual - predicted) < eps:
#                 predicted = actual
        # if verbose > 1 and actual != predicted:
        #     print(
        #         "{} {} vs {}::{} for query {}".format(
        #             i, actual, p, predicted, test_query
        #         )
        #     )

    # print("Accuracy {}".format(get_confusion_matrix(model, clevr_dataset_test, verbose=1).accuracy()))


#     if len(answer) > 0:
#         answer = answer[0][4:-1]
#     else:
#         answer = "invalid"

#     if answer == "true":
#         answer = "yes"
#     elif answer == "false":
#         answer = "no"
#     else:
#         answer = str(answer)

#     if answer == str(q['answer']):
#         correct += 1
#     elif answer == "invalid":
#         invalid += 1
#     else:
#         incorrect += 1

    # total += 1
    

# print(f"Correct: {correct}/{total} ({correct / total * 100:.2f})")
# print(f"Incorrect: {incorrect}/{total} ({incorrect / total * 100:.2f})")
# print(f"Invalid: {invalid}/{total} ({invalid / total * 100:.2f})")
