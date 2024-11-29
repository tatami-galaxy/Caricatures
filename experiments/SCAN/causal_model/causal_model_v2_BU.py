import copy
from tqdm.auto import tqdm
from datasets import load_dataset

# Bottom Up CAM 

# 1. Resolve U: Split sentence into lexical items, identify and interpret all verbs.
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
conjs = ['and', 'after', placeholder]

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


def causal_model(command):

    # Step 0: Split the command into lexical items (words)
    l0 = command.split()
    
    # STEP 1. Resolve U: Iidentify and interpret all verbs.
    # resolve verbs
    l1 =  [verbs[l] if l in verbs else l for l in l0]

    # STEP 2. Resolve D: Identify and interpret left/right/turn left/turn right
    l2 = [directions[l] if l in directions else l for l in l1]

    # STEP 3. Resolve V: Identify and interpret opposite/around
    l3 = [around_opposite[l] if l in around_opposite else l for l in l2]
    # opposite/around are lists
    oa_indices = [i for i in range(len(l3)) if isinstance(l3[i], list)]
    for oa_index in oa_indices:
        verb = l3[oa_index-1]
        direction = l3[oa_index+1]
        l3[oa_index] = [direction if i == 'direction' else verb for i in l3[oa_index]]
    # subsume turns and verbs
    del_indices = []
    for oa_index in reversed(oa_indices):
        del_indices.append(oa_index-1)
        del_indices.append(oa_index+1)
    l3 = [i for j, i in enumerate(l3) if j not in del_indices]

    # STEP 4. Resolve S: Identify and interpret twice/thrice
    # TODO
    l4 = copy.copy(l3)
    # find nums
    num_indices = [i for i in range(len(l3)) if not isinstance(l3[i], list) and l3[i] in nums]
    # repeat element to its left
    #for n in reversed(num_indices):
        # repeat element to its left

    print(l4)
    quit()

    # Remove placeholders
    l6 = []
    for l in l5:
        if len(l) == 0: continue
        for nl in l:
            item = placeholder.join(nl)
            l6.append(item)
    action = placeholder.join(l6)
    action = action.split(placeholder)
    action = [a for a in action if a != '']
    action = ' '.join(action)

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
    #command = 'look <empty> twice and jump opposite left twice'
    # command = 'turn <empty> left twice and jump <empty> <empty> <empty>'
    #command = 'run opposite left <empty> after walk <empty> right <empty>'
    # command = 'turn around right twice after run around right thrice'
    # command = 'walk opposite left <empty> <empty> <empty> <empty> <empty> <empty>'

    causal_model(command)
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
