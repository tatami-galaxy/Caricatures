from datasets import load_dataset
from pyvene import CausalModel
import itertools
from tqdm.auto import tqdm

# Bottom Up CAM 

# 1. Resolve U: Identify and interpret all verbs.
# 2. Resolve D: Identify and interpret left/right/turn left/turn right
# 3. Resolve V: Identify and interpret opposite/around
# 4. Resolve S: Identify and interpret twice/thrice
# 5. Resolve C: Identify and interpret and/after.


# utility functions
def rem_dups(l):
    l_un = []
    for item in l:
        if l not in l_un:
            l_un.append(item)
    return l_un

# https://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python
def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def add_empty_token(x):
    command_str = x['commands']
    command = command_str.split()
    padded_command = []
    index = 0
    c = 0
    while index < MAX_COMMAND_LEN:
        expected_cs = command_structure[index]
        if c < len(command) and command[c] in expected_cs:
            padded_command.append(command[c])
            c += 1
        else:
            padded_command.append(placeholder)
        index += 1

    x[input_column] = ' '.join(padded_command)
    return x


# Longest command is 9 words : https://arxiv.org/pdf/1711.00350
MAX_COMMAND_LEN = 9
placeholder = '<empty>'

VERBS = {
    'jump': 'I_JUMP',
    'look': 'I_LOOK',
    'walk': 'I_WALK',
    'run': 'I_RUN'
}
DIRECTIONS = {
    'left': 'I_TURN_LEFT',
    'right': 'I_TURN_RIGHT',
}
AROUND_OPPOSITE = {
    'around': ['direction', 'action']*4,
    'opposite': ['direction', 'direction', 'action'],
}
NUMS = {
    'twice': 2,
    'thrice': 3,
}
CONJS = ['and', 'after']

# command structure
command_structure = {
    0: VERBS,
    1: AROUND_OPPOSITE,
    2: DIRECTIONS,
    3: NUMS,
    4: CONJS,
    5: VERBS,  # 0
    6: AROUND_OPPOSITE,  # 1
    7: DIRECTIONS,  # 2
    8: NUMS,  # 3
}


### VARIABLES ###

LEAVES = [
    "verb1", "ar_op1", "dir1", "num1",
    "conj",
    "verb2", "ar_op2", "dir2", "num2"
]
NON_LEAVES = ["verb_res", "dir_res", "ar_op_res", "num_res", "conj_res"]
VARIABLES = LEAVES + NON_LEAVES


### FUNCTIONS ###

def verb_resolution(verb1, verb2):
    verb1 = VERBS[verb1] if verb1 in VERBS else verb1
    verb2 = VERBS[verb2] if verb2 in VERBS else verb2
    return [verb1, verb2]

def direction_resolution(dir1, dir2):
    dir1 = DIRECTIONS[dir1] if dir1 in DIRECTIONS else dir1
    dir2 = DIRECTIONS[dir2] if dir2 in DIRECTIONS else dir2
    return [dir1, dir2]

def around_opposite_resolution(verb_res, ar_op1, ar_op2, dir_res):
    # reformat
    items = [[verb_res[0], ar_op1, dir_res[0]], [verb_res[1], ar_op2, dir_res[1]]]
    res_items = []
    for item in items:
        # get verb, ar_op, direction
        verb = item[0]
        direction = item[2]
        ar_op = item[1]
        # resolve
        if ar_op == placeholder:
            res_items.append([direction, verb])
        else:
            res_item = AROUND_OPPOSITE[ar_op]
            res_items.append([direction if i == 'direction' else verb for i in res_item])
    return res_items

def num_resolution(ar_op_res, num1, num2):
    res_items = []
    # resolve num1
    if num1 != placeholder:
        res_items.append(ar_op_res[0]*NUMS[num1])
    else:
        res_items.append(ar_op_res[0])
    # resolve num2
    if num2 != placeholder:
        res_items.append(ar_op_res[1]*NUMS[num2])
    else:
        res_items.append(ar_op_res[1])
    return res_items

def conj_resolution(num_res, conj):
    if conj == 'and':
        output =  num_res[0] + num_res[1]
    elif conj == 'after':
        output = num_res[1] + num_res[0]
    else:
        output = num_res[0]
    # flatten?
    output = [l for l in output if l != placeholder]
    return ' '.join(output)


FUNCTIONS = {

    # leaves
    "verb1": lambda x: x, 
    "ar_op1": lambda x: x,
    "dir1": lambda x: x,
    "num1": lambda x: x,
    "conj": lambda x: x,
    "verb2": lambda x: x,
    "ar_op2": lambda x: x,
    "dir2": lambda x: x,
    "num2": lambda x: x,

    # resolve verbs
    "verb_res": verb_resolution,

    # resolve directions
    "dir_res": direction_resolution,

    # resolve around/opposite
    "ar_op_res": around_opposite_resolution,

    # resolve num
    "num_res": num_resolution,

    # resolve conjugation
    "conj_res": conj_resolution,
}


### VALUES (output) ###

VALUES = dict()

# leaves all values
VALUES["verb1"] = list(VERBS.keys())
VALUES["verb2"] = list(VERBS.keys())
VALUES["ar_op1"] = list(AROUND_OPPOSITE.keys())
VALUES["ar_op2"] = list(AROUND_OPPOSITE.keys())
VALUES["dir1"] = list(DIRECTIONS.keys())
VALUES["dir2"] = list(DIRECTIONS.keys())
VALUES["num1"] = list(NUMS.keys())
VALUES["num2"] = list(NUMS.keys())
VALUES["conj"] = CONJS

# verb_res all values
all_verbs = list(itertools.product(VALUES["verb1"], VALUES["verb2"]))
VALUES["verb_res"] = rem_dups([verb_resolution(tup[0], tup[1]) for tup in all_verbs])

# direction resolution all values
all_dirs = list(itertools.product(VALUES["dir1"], VALUES["dir2"]))
VALUES["dir_res"] = rem_dups([direction_resolution(tup[0], tup[1]) for tup in all_dirs])

# around/opposite resolution all values
all_ar_op = list(itertools.product(VALUES["verb_res"], VALUES["ar_op1"], VALUES["ar_op2"], VALUES["dir_res"]))
VALUES["ar_op_res"] = rem_dups([around_opposite_resolution(tup[0], tup[1], tup[2], tup[3]) for tup in all_ar_op])

# num resolution all values
all_nums = list(itertools.product(VALUES["ar_op_res"], VALUES["num1"], VALUES["num2"]))
VALUES["num_res"] = rem_dups([num_resolution(tup[0], tup[1], tup[2]) for tup in all_nums])

# conj resolution all values
all_conj = list(itertools.product(VALUES["num_res"], VALUES["conj"]))
VALUES["conj_res"] = list(set([conj_resolution(tup[0], tup[1]) for tup in all_conj]))


### PARENTS ###

PARENTS = {v:[] for v in VARIABLES}

PARENTS["verb_res"] = ["verb1", "verb2"]
PARENTS["dir_res"] = ["dir1", "dir2"]
PARENTS["ar_op_res"] = ["verb_res", "ar_op1", "ar_op2", "dir_res"]
PARENTS["num_res"] = ["ar_op_res", "num1", "num2"]
PARENTS["conj_res"] = ["num_res", "conj"]


### POSITIONS ###

# a dictionary with nodes as keys and positions as values
POS = {
    "verb1": (0.5, 0),
    "ar_op1": (1, 0),
    "dir1": (1.5, 0),
    "num1": (2, 0),
    "conj": (2.5, 0),
    "verb2": (3, 0),
    "ar_op2": (3.5, 0),
    "dir2": (4, 0),
    "num2": (4.5, 0),
    "verb_res": (1, 1),
    "dir_res": (3, 1),
    "ar_op_res": (2, 2),
    "num_res": (2.5, 3),
    "conj_res": (3, 4),
}


if __name__ == '__main__':

    scan_simple = load_dataset('scan', 'simple', trust_remote_code=True)
    scan_length = load_dataset('scan', 'length', trust_remote_code=True)

    simple_train = scan_simple['train']
    simple_test = scan_simple['test']
    length_train = scan_length['train']
    length_test = scan_length['test']

    data_splits = [simple_train, simple_test, length_train, length_test]

    causal_model = CausalModel(VARIABLES, VALUES, PARENTS, FUNCTIONS, pos=POS)
    #causal_model.print_structure()
    #print("Timesteps:", causal_model.timesteps)
    #quit()

    # filter out turns
    datasets = []
    for dataset in data_splits:
        dataset = dataset.filter(lambda x: 'turn' not in x["commands"].split())
        datasets.append(dataset)
    total_len = sum([len(d) for d in datasets])

    accuracy = 0
    bar = tqdm(range(total_len))
    for dataset in datasets:
        column_names = dataset.column_names
        input_column = column_names[0]

        # pad with placeholder
        dataset = dataset.map(
            add_empty_token,
            batched=False,
            # desc="Running tokenizer on dataset",
        )
        for x in dataset:
            label = x['actions']
            command_str = x['commands']
            command = command_str.split()
            causal_model_inputs = {LEAVES[i]:command[i] for i in range(MAX_COMMAND_LEN)}
            setting = causal_model.run_forward(causal_model_inputs)
            #print(setting['conj_res'])
            #print(label)
            #quit()
            if label==setting['conj_res']:
                accuracy += 1
            else:
                print(command_str)
                print(setting['conj_res'])
                quit()
            bar.update(1)

    print('accuracy on simple and length splits : {}'.format(accuracy/total_len))
