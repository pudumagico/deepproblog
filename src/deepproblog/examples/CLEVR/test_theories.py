import json
import sys
sys.path.append('../../')

from translate import func_to_asp, json_to_facts, parse_facts
from tqdm import tqdm
import sys

from pyswip import Prolog
from problog.program import PrologString
from problog.core import ProbLog
from problog import get_evaluatable
import clingo

with open("CLEVR_v1.0/questions/CLEVR_val_questions.json") as fp:
    questions = json.load(fp)["questions"]



with open("prolog_theory.lp", "r") as fp:
    prolog_theory = fp.read()
with open("problog_theory.lp", "r") as fp:
    problog_theory = fp.read()
with open("asp_theory.lp", "r") as fp:
    asp_theory = fp.read()

facts = json_to_facts('./scene_encoding_det_epoch200_conf25.json')

prolog = Prolog()


######################### Prolog ############################

print('TESTING THEORY FOR PROLOG')

correct = 0
incorrect = 0
invalid = 0
total = 0


for q in questions:

    img_index = str(q['image_index'])
    incumbent_facts = parse_facts(facts, img_index)
    
    program = prolog_theory
    program += '\n'
    program += func_to_asp(q["program"]) 
    program += '\n'
    program += incumbent_facts
    program += '\n'
    
    
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

    if str(correct_answer) == str(prolog_answer):
        correct+=1
    else:
        # print(q)
        # print(correct_answer, prolog_answer)
        incorrect+=1

        
        # print(total)
    total+=1
    
    if total == 2000:
        break

print(f"Correct: {correct}/{total} ({correct / total * 100:.2f})")
print(f"Incorrect: {incorrect}/{total} ({incorrect / total * 100:.2f})")
print(f"Invalid: {invalid}/{total} ({invalid / total * 100:.2f})")


######################### Problog ############################

print('TESTING THEORY FOR PROBLOG')


correct = 0
incorrect = 0
invalid = 0
total = 0


for q in questions:

    img_index = str(q['image_index'])
    incumbent_facts = parse_facts(facts, img_index)
    
    program = problog_theory
    program += '\n'
    program += func_to_asp(q["program"]) 
    program += '\n'
    program += incumbent_facts
    program += '\n'
    program += 'query(ans(X)).'
    
    problog_answer = get_evaluatable().create_from(program).evaluate()
    
    correct_answer = q['answer']
    if correct_answer == 'no':
        correct_answer = 'false'
    if correct_answer == 'yes':
        correct_answer = 'true'
    

    problog_answer =  str(list(problog_answer.keys())[0]).split('(')[1].split(')')[0]
    

    if str(correct_answer) == str(problog_answer):
        correct+=1
    else:
        # print(q)
        # print(correct_answer, problog_answer)
        incorrect+=1
        # print(func_to_asp(q["program"]))
        # print(incumbent_facts)
        # print(total)
    total+=1

    if total == 2000:
        break

print(f"Correct: {correct}/{total} ({correct / total * 100:.2f})")
print(f"Incorrect: {incorrect}/{total} ({incorrect / total * 100:.2f})")
print(f"Invalid: {invalid}/{total} ({invalid / total * 100:.2f})")




# ######################### ASP ############################

print('TESTING THEORY FOR ASP')

correct = 0
incorrect = 0
invalid = 0
total = 0


for q in questions:

    img_index = str(q['image_index'])
    incumbent_facts = parse_facts(facts, img_index)
    
    program = asp_theory
    program += '\n'
    program += func_to_asp(q["program"]) 
    program += '\n'
    program += incumbent_facts

    
    correct_answer = q['answer']
    if correct_answer == 'no':
        correct_answer = 'false'
    if correct_answer == 'yes':
        correct_answer = 'true'
    
    
    ctl = clingo.Control()
    ctl.add("base", [], program)
    ctl.ground([("base", [])])
    # ctl.solve(on_model=on_model(x))
    # print(x)
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            if model.symbols(shown=True):
                asp_answer = str(model.symbols(shown=True)[0]).split('(')[1].split(')')[0]
            
           
    if str(correct_answer) == str(asp_answer):
        correct+=1
    else:
        # print(q)
        # print(correct_answer, asp_answer)
        incorrect+=1
        # print(total)
    total+=1

    if total == 2000:
        break

print(f"Correct: {correct}/{total} ({correct / total * 100:.2f})")
print(f"Incorrect: {incorrect}/{total} ({incorrect / total * 100:.2f})")
print(f"Invalid: {invalid}/{total} ({invalid / total * 100:.2f})")

