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


from deepproblog.heuristics import geometric_mean
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.utils import get_configuration, format_time_precise, config_to_string
from connect import ClevrDataset, ClevrDatasetStateDescription


with open("CLEVR_v1.0/questions/CLEVR_val_questions.json") as fp:
    questions = json.load(fp)["questions"]

dprogram = r'''
    nn(label, [1], (I,B) , [obj(B,cylinder,large,gray,metal,X1,Y1,X2,Y2), obj(B,sphere,large,gray,metal,X1,Y1,X2,Y2), obj(B,cube,large,gray,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,gray,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,gray,rubber,X1,Y1,X2,Y2), obj(B,cube,large,gray,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,blue,metal,X1,Y1,X2,Y2), obj(B,sphere,large,blue,metal,X1,Y1,X2,Y2), obj(B,cube,large,blue,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,blue,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,blue,rubber,X1,Y1,X2,Y2), obj(B,cube,large,blue,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,brown,metal,X1,Y1,X2,Y2), obj(B,sphere,large,brown,metal,X1,Y1,X2,Y2), obj(B,cube,large,brown,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,brown,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,brown,rubber,X1,Y1,X2,Y2), obj(B,cube,large,brown,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,yellow,metal,X1,Y1,X2,Y2), obj(B,sphere,large,yellow,metal,X1,Y1,X2,Y2), obj(B,cube,large,yellow,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,yellow,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,yellow,rubber,X1,Y1,X2,Y2), obj(B,cube,large,yellow,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,red,metal,X1,Y1,X2,Y2), obj(B,sphere,large,red,metal,X1,Y1,X2,Y2), obj(B,cube,large,red,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,red,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,red,rubber,X1,Y1,X2,Y2), obj(B,cube,large,red,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,green,metal,X1,Y1,X2,Y2), obj(B,sphere,large,green,metal,X1,Y1,X2,Y2), obj(B,cube,large,green,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,green,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,green,rubber,X1,Y1,X2,Y2), obj(B,cube,large,green,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,purple,metal,X1,Y1,X2,Y2), obj(B,sphere,large,purple,metal,X1,Y1,X2,Y2), obj(B,cube,large,purple,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,purple,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,purple,rubber,X1,Y1,X2,Y2), obj(B,cube,large,purple,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,cyan,metal,X1,Y1,X2,Y2), obj(B,sphere,large,cyan,metal,X1,Y1,X2,Y2), obj(B,cube,large,cyan,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,cyan,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,cyan,rubber,X1,Y1,X2,Y2), obj(B,cube,large,cyan,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,gray,metal,X1,Y1,X2,Y2), obj(B,sphere,small,gray,metal,X1,Y1,X2,Y2), obj(B,cube,small,gray,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,gray,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,gray,rubber,X1,Y1,X2,Y2), obj(B,cube,small,gray,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,blue,metal,X1,Y1,X2,Y2), obj(B,sphere,small,blue,metal,X1,Y1,X2,Y2), obj(B,cube,small,blue,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,blue,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,blue,rubber,X1,Y1,X2,Y2), obj(B,cube,small,blue,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,brown,metal,X1,Y1,X2,Y2), obj(B,sphere,small,brown,metal,X1,Y1,X2,Y2), obj(B,cube,small,brown,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,brown,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,brown,rubber,X1,Y1,X2,Y2), obj(B,cube,small,brown,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,yellow,metal,X1,Y1,X2,Y2), obj(B,sphere,small,yellow,metal,X1,Y1,X2,Y2), obj(B,cube,small,yellow,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,yellow,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,yellow,rubber,X1,Y1,X2,Y2), obj(B,cube,small,yellow,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,red,metal,X1,Y1,X2,Y2), obj(B,sphere,small,red,metal,X1,Y1,X2,Y2), obj(B,cube,small,red,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,red,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,red,rubber,X1,Y1,X2,Y2), obj(B,cube,small,red,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,green,metal,X1,Y1,X2,Y2), obj(B,sphere,small,green,metal,X1,Y1,X2,Y2), obj(B,cube,small,green,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,green,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,green,rubber,X1,Y1,X2,Y2), obj(B,cube,small,green,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,purple,metal,X1,Y1,X2,Y2), obj(B,sphere,small,purple,metal,X1,Y1,X2,Y2), obj(B,cube,small,purple,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,purple,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,purple,rubber,X1,Y1,X2,Y2), obj(B,cube,small,purple,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,cyan,metal,X1,Y1,X2,Y2), obj(B,sphere,small,cyan,metal,X1,Y1,X2,Y2), obj(B,cube,small,cyan,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,cyan,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,cyan,rubber,X1,Y1,X2,Y2), obj(B,cube,small,cyan,rubber,X1,Y1,X2,Y2)]) :: box(I,B,X1,Y1,X2,Y2).
    '''

termPath = f'img ./CLEVR_v1.0/images/val'
img_size = 480
domain = ['LaGraMeCy', 'LaGraMeSp', 'LaGraMeCu', 'LaGraRuCy', 'LaGraRuSp', 'LaGraRuCu', 'LaBlMeCy', 'LaBlMeSp', 'LaBlMeCu', 'LaBlRuCy',
          'LaBlRuSp', 'LaBlRuCu', 'LaBrMeCy', 'LaBrMeSp', 'LaBrMeCu', 'LaBrRuCy', 'LaBrRuSp', 'LaBrRuCu', 'LaYeMeCy', 'LaYeMeSp',
          'LaYeMeCu', 'LaYeRuCy', 'LaYeRuSp', 'LaYeRuCu', 'LaReMeCy', 'LaReMeSp', 'LaReMeCu', 'LaReRuCy', 'LaReRuSp', 'LaReRuCu',
          'LaGreMeCy', 'LaGreMeSp', 'LaGreMeCu', 'LaGreRuCy', 'LaGreRuSp', 'LaGreRuCu', 'LaPuMeCy', 'LaPuMeSp', 'LaPuMeCu', 'LaPuRuCy',
          'LaPuRuSp', 'LaPuRuCu', 'LaCyMeCy', 'LaCyMeSp', 'LaCyMeCu', 'LaCyRuCy', 'LaCyRuSp', 'LaCyRuCu', 'SmGraMeCy', 'SmGraMeSp',
          'SmGraMeCu', 'SmGraRuCy', 'SmGraRuSp', 'SmGraRuCu', 'SmBlMeCy', 'SmBlMeSp', 'SmBlMeCu', 'SmBlRuCy', 'SmBlRuSp', 'SmBlRuCu',
          'SmBrMeCy', 'SmBrMeSp', 'SmBrMeCu', 'SmBrRuCy', 'SmBrRuSp', 'SmBrRuCu', 'SmYeMeCy', 'SmYeMeSp', 'SmYeMeCu', 'SmYeRuCy',
          'SmYeRuSp', 'SmYeRuCu', 'SmReMeCy', 'SmReMeSp', 'SmReMeCu', 'SmReRuCy', 'SmReRuSp', 'SmReRuCu', 'SmGreMeCy', 'SmGreMeSp',
          'SmGreMeCu', 'SmGreRuCy', 'SmGreRuSp', 'SmGreRuCu', 'SmPuMeCy', 'SmPuMeSp', 'SmPuMeCu', 'SmPuRuCy', 'SmPuRuSp', 'SmPuRuCu',
          'SmCyMeCy', 'SmCyMeSp', 'SmCyMeCu', 'SmCyRuCy', 'SmCyRuSp', 'SmCyRuCu']


filename1 = 'factsList.pk'
filename2 = 'dataList.pk'

factsList = ''
dataList = ''

with open(filename1, 'rb') as fi:
    factsList = pickle.load(fi)

with open(filename2, 'rb') as fi:
    dataList = pickle.load(fi)
if dataList:
    pass
else:
    factsList, dataList = termPath2dataList(termPath, img_size, domain)
    with open(filename1, 'wb') as fi:
        pickle.dump(factsList, fi)
    with open(filename2, 'wb') as fi:
        pickle.dump(dataList, fi)

correct = 0
incorrect = 0
invalid = 0
total = 0

questionCounter = 0

with open("program.lp", "r") as fp:
    theory = fp.read()

i = int(sys.argv[1]) if len(sys.argv) > 1 else 0

parameters = {
    "method": ["gm", "exact"],
    "N": [1, 2, 3],
    "pretrain": [0],
    "exploration": [False, True],
    "run": range(5),
}

configuration = get_configuration(parameters, i)
torch.manual_seed(configuration["run"])

name = "clevr_" + config_to_string(configuration) + "_" + format_time_precise()
print(name)

dictionaries = utils.build_dictionaries('./CLEVR_v1.0')
# clevr_dataset_train, clevr_dataset_test  = initialize_dataset('./CLEVR_v1.0', dictionaries, '')

clevr_dataset_test = ClevrDataset('./CLEVR_v1.0', False, dictionaries)
network = Net()
net = Network(network, "label", batching=False)

facts = json_to_facts('./scene_encoding_det_epoch200_conf25.json')


for q in tqdm(questions):
    # print(q)
    #if questionCounter < 86300:
    #    questionCounter += 1
    #    continue
    img_index = str(q['image_index'])
    # print(img_index)
    if int(img_index) == 0:
        # print('continuing')
        continue
    incumbent_facts = parse_facts(facts, img_index)
    aspProgram = func_to_asp(q["program"])
    # print(aspProgram)
    aspProgram += theory 
    # aspProgram += '\n'
    # aspProgram += incumbent_facts
    # print(incumbent_facts)
    # print(aspProgram)
    # print(facts)


    m = Net()
    nnMapping = {'label': m}

    model = Model(aspProgram, [net], load=False)

    # model.set_engine(ExactEngine(model), cache=True)
    # model.set_engine(
    #     ApproximateEngine(
    #         model, 1, geometric_mean, exploration=configuration["exploration"]
    #     )
    # )
    model.set_engine(ExactEngine(model), cache=True)

    # loader = DataLoader(clevr_dataset_test, 1, False)

    print("Accuracy {}".format(get_confusion_matrix(model, clevr_dataset_test, verbose=1).accuracy()))

    # models = NeurASPobj.infer(dataDic=dataList[q['image_index']], obs='', mvpp=aspProgram + facts)

    # answer = [atom for atom in models[0] if re.search(r"ans\(.*\)", atom)]

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

#     total += 1

# print(f"Correct: {correct}/{total} ({correct / total * 100:.2f})")
# print(f"Incorrect: {incorrect}/{total} ({incorrect / total * 100:.2f})")
# print(f"Invalid: {invalid}/{total} ({invalid / total * 100:.2f})")
