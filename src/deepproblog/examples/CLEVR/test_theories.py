import json
import sys
sys.path.append('../../')

from translate import func_to_asp, json_to_facts, parse_facts
from tqdm import tqdm
import sys

from pyswip import Prolog
import clingo

with open("CLEVR_v1.0/questions/CLEVR_val_questions.json") as fp:
    questions = json.load(fp)["questions"]

correct = 0
incorrect = 0
invalid = 0
total = 0

with open("prolog_theory.lp", "r") as fp:
    theory = fp.read()

facts = json_to_facts('./scene_encoding_det_epoch200_conf25.json')

prolog = Prolog()
clingo = clingo.Control()

for q in tqdm(questions):

    img_index = str(q['image_index'])
    incumbent_facts = parse_facts(facts, img_index)
    
    program = theory
    program += '\n'
    program += func_to_asp(q["program"]) 
    program += '\n'
    program += incumbent_facts
    program += '\n'
    
    print(q)
    print(incumbent_facts)
    print(func_to_asp(q["program"])) 
    
    
    with open("incumbent_program.lp", "w") as ip:
        ip.write(program)
    ip.close()
    
    correct_answer = q['answer']
    if correct_answer == 'no':
        correct_answer = 'false'
    if correct_answer == 'yes':
        correct_answer = 'true'
    
    prolog.consult('incumbent_program.lp')
    
    if list(prolog.query('ans(X)')):
        prolog_answer = list(prolog.query('ans(X)'))[0]['X']
    else:
        prolog_answer = None
    # print(correct_answer, prolog_answer)

    if str(correct_answer) == str(prolog_answer):
        correct+=1
    else:
        
        incorrect+=1
    total+=1
    
    if total == 10:
        break

print(f"Correct: {correct}/{total} ({correct / total * 100:.2f})")
print(f"Incorrect: {incorrect}/{total} ({incorrect / total * 100:.2f})")
print(f"Invalid: {invalid}/{total} ({invalid / total * 100:.2f})")
