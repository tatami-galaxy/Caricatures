import copy
from tqdm.auto import tqdm
from itertools import chain
from datasets import load_dataset

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


# https://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python
def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def causal_model(command):

    # Step 0: Split the command into lexical items (words)
    l0 = command.split()
    
    # STEP 1. Resolve U: Iidentify and interpret all verbs.
    # resolve verbs
    l1 =  [verbs[l] if l in verbs else l for l in l0]

    print(l1)

    # STEP 2. Resolve D: Identify and interpret left/right/turn left/turn right
    l2 = [directions[l] if l in directions else l for l in l1]

    print(l2)

    # STEP 3. Resolve V: Identify and interpret opposite/around
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

    print(l2)

    # subsume turns and verbs
    del_indices = []
    for oa_index in reversed(oa_indices):
        del_indices.append(oa_index-1)
        del_indices.append(oa_index+1)
    l3 = [i for j, i in enumerate(l2) if j not in del_indices]

    print(l3)

    # STEP 4. Resolve S: Identify and interpret twice/thrice
    l4 = copy.copy(l3)
    # find nums
    num_indices = [i for i in range(len(l3)) if not isinstance(l3[i], list) and l3[i] in nums]
    # repeat elements to its left or list to its left
    for n in reversed(num_indices):
        num = l3[n]
        l4[n] = [l3[n-1]]*nums[num]
        del l4[n-1]

    print(l4)

    # STEP 5. Resolve C: Identify and interpret and/after
    l5 = []
    # find and/after
    conj_index = [i for i in range(len(l4)) if not isinstance(l4[i], list) and l4[i] in conjs]
    if len(conj_index) > 0:
        conj = l4[conj_index[0]]
        conj_index = conj_index[0]
    else:
        conj = placeholder
    if conj == 'after':
        l5.extend(l4[conj_index+1:])
        l5.extend(l4[:conj_index])
    elif conj == 'and':
        l5.extend(l4[:conj_index])
        l5.extend(l4[conj_index+1:])
    else:
        l5 = copy.copy(l4)

    print(l5)
    quit()

    # Flatten and remove placeholders
    l6 = list(flatten(l5))
    l6 = [l for l in l6 if l != placeholder]
    action = ' '.join(l6)

    return action


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


if __name__ == '__main__':

    ## testing ##

    command = 'look around right twice and jump opposite left thrice'
    #command = 'look <empty> <empty> twice and jump opposite left <empty>'
    #command = 'run opposite left <empty> after walk <empty> right <empty>'
    #command = 'look <empty> <empty> twice <empty> <empty> <empty> <empty> <empty>'
    #command = 'walk <empty> <empty> <empty> after run around right twice'
    #command = 'walk opposite left twice and walk opposite right thrice'

    print(causal_model(command))
    quit()

    ## testing end ##

    scan_simple = load_dataset('scan', 'simple', trust_remote_code=True)
    scan_length = load_dataset('scan', 'length', trust_remote_code=True)

    simple_train = scan_simple['train']
    simple_test = scan_simple['test']
    length_train = scan_length['train']
    length_test = scan_length['test']
    data_splits = [simple_train, simple_test, length_train, length_test]

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
        for example in dataset:
            output = causal_model(example['commands'])
            label = example['actions']
            if label == output:
                accuracy += 1
            else:
                print(example['commands'])
                print(label)
                print(output)
                quit()
            bar.update(1)

    print('accuracy on simple and length splits : {}'.format(accuracy/total_len))
