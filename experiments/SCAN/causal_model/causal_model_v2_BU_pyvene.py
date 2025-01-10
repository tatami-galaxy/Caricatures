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


### VARIABLES ###

leaves = [
    "verb1", "ar_op1", "dir1", "num1",
    "conj",
    "verb2", "ar_op2", "dir2", "num2"
]
non_leaves = ["verb_res", "dir_res", "ar_op_res", "num_res", "conj_res"]
variables = leaves + non_leaves


# TODO:
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
    # 'I_LOOK', 'around', 'I_TURN_RIGHT', 'twice', 'and', 'I_JUMP', 'opposite', 'I_TURN_LEFT', 'thrice'
    items = [[verb_res[0], ar_op1, dir_res[0]], [verb_res[1], ar_op2, dir_res[1]]]
    for item in items:

        ##
        oa_indices = [1, 6]
        #l3 = [around_opposite[l] if l in around_opposite else l for l in l2]
        # opposite/around are lists
        for oa_index in oa_indices:
            verb = l2[oa_index-1]
            direction = l2[oa_index+1]
            if l2[oa_index] == placeholder:
                l2[oa_index] = [direction, verb]
            else:
                l2[oa_index] = around_opposite[l2[oa_index]]
                l2[oa_index] = [direction if i == 'direction' else verb for i in l2[oa_index]]
        # subsume turns and verbs
        del_indices = []
        for oa_index in reversed(oa_indices):
            del_indices.append(oa_index-1)
            del_indices.append(oa_index+1)
        l3 = [i for j, i in enumerate(l2) if j not in del_indices]
        ##



def arr_opp_resolution():
    pass

def resolve_num():
    pass

def conj_resolution():
    pass


functions = {

    # leaves
    "verb1": lambda x: x, 
    "ar_op1": lambda x: x,
    "dir1": lambda x: x,
    "num1": lambda x: x,
    "conj",: lambda x: x,
    "verb2": lambda x: x,
    "ar_op2": lambda x: x,
    "dir2": lambda x: x,
    "num2": lambda x: x,

    # resolve verbs
    "verb_res": verb_resolution,

    # resolve directions
    "dir_res": direction_resolution,

    # resolve around/opposite
    "ar_op_res": arr_opp_resolution,

    # resolve num
    "num_res": resolve_num,

    # resolve conjugation
    "conj_res": conj_resolution,
}


### VALUES ###

values = dict()

# leaves
values["act1"] = list(actions.keys())
values["act2"] = list(actions.keys())
values["trn1"] = list(turns.keys())
values["trn2"] = list(turns.keys())
values["dir1"] = list(directions.keys())
values["dir2"] = list(directions.keys())
values["num1"] = list(nums.keys())
values["num2"] = list(nums.keys())
values["conj"] = conjs


# resolve turn
values["trn1_res"] = [resolve_turn(t) for t in values["trn1"]]
values["trn2_res"] = values["trn1_res"].copy()

# turn+dir
all_trn_dir = list(itertools.product(values["trn1_res"], values["dir1"]))
values["trn1_dir1"] = list(set([turn_function(tup[0], tup[1]) for tup in all_trn_dir]))
values["trn2_dir2"] = values["trn1_dir1"].copy()

# act+turn_dir
all_act_trn_dir = list(itertools.product(values["act1"], values["trn1_dir1"]))
values["act1_trn1_dir1"] = list(set([action_function(tup[0], tup[1]) for tup in all_act_trn_dir]))
values["act2_trn2_dir2"] = values["act1_trn1_dir1"].copy()

# resolve num
values["num1_res"] = [resolve_num(n) for n in values["num1"]]
values["num2_res"] = values["num1_res"].copy()

# act_turn_dir+num
all_act_trn_dir_num = list(itertools.product(values["act1_trn1_dir1"], values["num1_res"]))
values["act1_trn1_dir1_num1"] = list(set([num_function(tup[0], tup[1]) for tup in all_act_trn_dir_num]))
values["act2_trn2_dir2_num2"] = values["act1_trn1_dir1_num1"].copy()

# conj_left
all_conj_left = list(itertools.product(values["act1_trn1_dir1_num1"], conjs))
values["conj_left"] = list(set([conjugation_left(tup[0], tup[1]) for tup in all_conj_left]))

# conj_right
all_conj_right = list(itertools.product(values["conj_left"], values["act2_trn2_dir2_num2"]))
values["conj_right"] = list(set([conjugation_right(tup[0], tup[1]) for tup in all_conj_right]))


### PARENTS ###

parents = {v:[] for v in variables}
# left subtree
parents["trn1_res"] = ["trn1"]
parents["trn1_dir1"] = ["trn1_res", "dir1"]
parents["act1_trn1_dir1"] = ["act1", "trn1_dir1"]
parents["num1_res"] = ["num1"]
parents["act1_trn1_dir1_num1"] = ["act1_trn1_dir1", "num1_res"]
# right subtree
parents["trn2_res"] = ["trn2"]
parents["trn2_dir2"] = ["trn2_res", "dir2"]
parents["act2_trn2_dir2"] = ["act2", "trn2_dir2"]
parents["num2_res"] = ["num2"]
parents["act2_trn2_dir2_num2"] = ["act2_trn2_dir2", "num2_res"]
# merge
parents["conj_left"] = ["act1_trn1_dir1_num1", "conj"]
parents["conj_right"] = ["conj_left", "act2_trn2_dir2_num2"]


### POSITIONS ###

# a dictionary with nodes as keys and positions as values
pos = {

    # left subtree
    "act1": (1, 0),
    "trn1": (2, 0),
    "dir1": (1.9, 0.05),
    "num1": (4, 0),
    "conj": (4.11, 0.1),
    "act1": (0.2, 0),
    "trn1": (1, 0.1),
    "trn1_res": (1.33, 0.4),
    "dir1": (2, 0.3),
    "num1": (2.8, 0),
    "num1_res": (3, 0.2),
    "trn1_dir1": (1.4, 0.96),
    "act1_trn1_dir1": (0.2, 1.5),
    "act1_trn1_dir1_num1": (2.5, 1.8),

    # right subtree
    "act2": (5, 0),
    "trn2": (6, 0),
    "dir2": (5.9, 0.05),
    "num2": (9, 0),
    "act2": (5.2, 0),
    "trn2": (6, 0.1),
    "trn2_res": (6, 0.4),
    "dir2": (7, 0.3),
    "num2": (9.8, 0),
    "num2_res": (9, 0.2),
    "trn2_dir2": (6.4, 1),
    "act2_trn2_dir2": (5.2, 1.5),
    "act2_trn2_dir2_num2": (7.5, 1.8),

    # merge
    "conj_left": (3.5, 2.2),
    "conj_right": (4.5, 3.2),
}


if __name__ == '__main__':

    scan_simple = load_dataset('scan', 'simple', trust_remote_code=True)
    scan_length = load_dataset('scan', 'length', trust_remote_code=True)

    simple_train = scan_simple['train']
    simple_test = scan_simple['test']
    length_train = scan_length['train']
    length_test = scan_length['test']

    data_splits = [simple_train, simple_test, length_train, length_test]
    total_len = sum([len(s) for s in data_splits])

    causal_model = CausalModel(variables, values, parents, functions, pos=pos)
    #causal_model.print_structure()
    #print("Timesteps:", causal_model.timesteps)
    #quit()

    accuracy = 0
    bar = tqdm(range(total_len))
    for dataset in data_splits:
        for x in dataset:
            command_str = x['commands']
            label = x['actions']
            
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
                    padded_command.append(EMPTY)
                index += 1

            causal_model_inputs = {leaves[i]:padded_command[i] for i in range(max_len)}
            setting = causal_model.run_forward(causal_model_inputs)
            if label==setting['conj_right']:
                accuracy += 1
            else:
                print(command_str)
                print(setting)
                quit()
            bar.update(1)

    print('accuracy on simple and length splits : {}'.format(accuracy/total_len))
