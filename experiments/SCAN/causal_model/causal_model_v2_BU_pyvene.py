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


# Longest command is 9 words : https://arxiv.org/pdf/1711.00350
max_len = 9
placeholder = '<empty>'
verbs = {
    'jump': 'I_JUMP',
    'look': 'I_LOOK',
    'walk': 'I_WALK',
    'run': 'I_RUN'
}
directions = {
    'left': 'I_TURN_LEFT',
    'right': 'I_TURN_RIGHT',
}
around_opposite = {
    'around': ['direction', 'action']*4,
    'opposite': ['direction', 'direction', 'action'],
}
nums = {
    'twice': 2,
    'thrice': 3,
}
conjs = ['and', 'after']

# command structure
command_structure = {
    0: verbs,
    1: around_opposite,
    2: directions,
    3: nums,
    4: conjs,
    5: verbs,  # 0
    6: around_opposite,  # 1
    7: directions,  # 2
    8: nums,  # 3
}

# util functions
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
    while index < max_len:
        expected_cs = command_structure[index]
        if c < len(command) and command[c] in expected_cs:
            padded_command.append(command[c])
            c += 1
        else:
            padded_command.append(placeholder)
        index += 1

    x[input_column] = ' '.join(padded_command)
    return x


### VARIABLES ###

leaves = [
    "verb1", "ar_op1", "dir1", "num1",
    "conj",
    "verb2", "ar_op2", "dir2", "num2"
]
non_leaves = ["verb_res", "dir_res", "ar_op_res", "num_res", "conj_res"]
variables = leaves + non_leaves


### FUNCTIONS ###

def verb_resolution(verb1, verb2):
    verb1 = verbs[verb1] if verb1 in verbs else verb1
    verb2 = verbs[verb2] if verb2 in verbs else verb2
    return [verb1, verb2]

def direction_resolution(dir1, dir2):
    dir1 = directions[dir1] if dir1 in directions else dir1
    dir2 = directions[dir2] if dir2 in directions else dir2
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
            res_item = around_opposite[ar_op]
            res_items.append([direction if i == 'direction' else verb for i in res_item])
    return res_items

def num_resolution(ar_op_res, num1, num2):
    res_items = []
    # resolve num1
    if num1 != placeholder:
        res_items.append(ar_op_res[0]*nums[num1])
    else:
        res_items.append(ar_op_res[0])
    # resolve num2
    if num2 != placeholder:
        res_items.append(ar_op_res[1]*nums[num2])
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


functions = {

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

values = dict()

# leaves all values
values["verb1"] = list(verbs.keys())
values["verb2"] = list(verbs.keys())
values["ar_op1"] = list(around_opposite.keys())
values["ar_op2"] = list(around_opposite.keys())
values["dir1"] = list(directions.keys())
values["dir2"] = list(directions.keys())
values["num1"] = list(nums.keys())
values["num2"] = list(nums.keys())
values["conj"] = conjs

# verb_res all values
all_verbs = list(itertools.product(values["verb1"], values["verb2"]))
values["verb_res"] = rem_dups([verb_resolution(tup[0], tup[1]) for tup in all_verbs])

# direction resolution all values
all_dirs = list(itertools.product(values["dir1"], values["dir2"]))
values["dir_res"] = rem_dups([direction_resolution(tup[0], tup[1]) for tup in all_dirs])

# around/opposite resolution all values
all_ar_op = list(itertools.product(values["verb_res"], values["ar_op1"], values["ar_op2"], values["dir_res"]))
values["ar_op_res"] = rem_dups([around_opposite_resolution(tup[0], tup[1], tup[2], tup[3]) for tup in all_ar_op])

# num resolution all values
all_nums = list(itertools.product(values["ar_op_res"], values["num1"], values["num2"]))
values["num_res"] = rem_dups([num_resolution(tup[0], tup[1], tup[2]) for tup in all_nums])

# conj resolution all values
all_conj = list(itertools.product(values["num_res"], values["conj"]))
values["conj_res"] = list(set([conj_resolution(tup[0], tup[1]) for tup in all_conj]))


### PARENTS ###

parents = {v:[] for v in variables}

parents["verb_res"] = ["verb1", "verb2"]
parents["dir_res"] = ["dir1", "dir2"]
parents["ar_op_res"] = ["verb_res", "ar_op1", "ar_op2", "dir_res"]
parents["num_res"] = ["ar_op_res", "num1", "num2"]
parents["conj_res"] = ["num_res", "conj"]


### POSITIONS ###

# a dictionary with nodes as keys and positions as values
pos = {
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

    causal_model = CausalModel(variables, values, parents, functions, pos=pos)
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
            causal_model_inputs = {leaves[i]:command[i] for i in range(max_len)}
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
