import copy
from tqdm.auto import tqdm
from datasets import load_dataset

# Longest command is 9 words : https://arxiv.org/pdf/1711.00350
max_len = 9
placeholder = '<empty>'
verbs = {
    'turn': placeholder,
    'jump': 'I_JUMP',
    'look': 'I_LOOK',
    'walk': 'I_WALK',
    'run': 'I_RUN'
}
directions = {
    'left': 'I_TURN_LEFT',
    'right': 'I_TURN_RIGHT',
    placeholder: placeholder
}
around_opposite = {
    'around': ['direction', 'action']*4,
    'opposite': ['direction', 'direction', 'action'],
    placeholder: ['direction', 'action']
}
nums = {
    'twice': 2,
    'thrice': 3,
    placeholder: 1,
}
conjs = ['and', 'after']

# command structure
command_structure = {
    0: verbs,
    1: around_opposite,
    2: directions,
    3: nums,
    4: conjs,
    5: verbs, # 0
    6: around_opposite, # 1
    7: directions, # 2
    8: nums, # 3
}


def causal_model(command):

    # Step 0: Split the command into lexical items (words)
    l0 = command.split()
    # STEP 1. Resolve C: Split based on conj
    conj = l0[4]
    if conj == 'and':
        # maintain order of command
        l11 = l0[:4] 
        l12 = l0[5:]
        # merge
        l1 = [l11, l12]
    elif conj == 'after':
        # reverse order of command
        l11 = l0[5:]
        l12 = l0[:4] 
        # merge
        l1 = [l11, l12]
    else:
        l1 = [l0[:4]]


    # STEP 2. Resolve S: Interpret twice/thrice for repetition as individual elements
    l2 = []
    for l in l1:
        # find nums and resolve
        num = l[-1]
        l_copy = [l[:-1]]*nums[num]
        l2.append(l_copy)


    # STEP 3: Resolve V: Interpret opposite/around and handle direction repeats
    l3 = []
    for l in l2:
        new_l = []
        # has nested lists
        for nl in l:
            # resolve around/opposite
            item = copy.copy(nl)
            ar_opp = nl[1]
            resl = around_opposite[ar_opp]
            item[1] = resl
            new_l.append(item)
        l3.append(new_l)


    # STEP 4: Resolve D: Identify and interpret directions
    l4 = []
    for l in l3:
        new_l = []
        # has nested lists
        for nl in l:
            item = copy.copy(nl)
            # get direction
            dir = nl[-1]
            # replace direction in around/opposite
            item[1] = [directions[dir] if i == 'direction' else i for i in nl[1]]
            del item[-1]
            new_l.append(item)
        l4.append(new_l)
            

    # STEP 5: Resolve U: Identify and replace all verbs
    l5 = []
    for l in l4:
        new_l = []
        # has nested lists
        for nl in l:
            item = copy.copy(nl)
            # get verb
            verb = nl[0]
            # replace direction in around/opposite
            item[1] = [verbs[verb] if i == 'action' else i for i in nl[1]]
            # only one term left
            del item[0]
            # remove unnecessary nesting
            item = item[0]
            new_l.append(item)
        l5.append(new_l)


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

    #command = 'look around right twice and turn opposite left twice'
    #command = 'turn <empty> left twice and jump <empty> <empty> <empty>'
    #command = 'run opposite left <empty> after walk <empty> right <empty>'
    #command = 'turn around right twice after run around right thrice'
    #command = 'walk opposite left <empty> <empty> <empty> <empty> <empty> <empty>'
    #print(causal_model(command))
    #quit()

    ## testing end ##

    scan_simple = load_dataset('scan', 'simple', trust_remote_code=True)
    scan_length = load_dataset('scan', 'length', trust_remote_code=True)

    simple_train = scan_simple['train']
    simple_test = scan_simple['test']
    length_train = scan_length['train']
    length_test = scan_length['test']

    data_splits = [simple_train, simple_test, length_train, length_test]
    total_len = sum([len(s) for s in data_splits])

    accuracy = 0
    bar = tqdm(range(total_len))
    for dataset in data_splits:
        column_names = dataset.column_names
        input_column = column_names[0]
        dataset = dataset.map(
            add_empty_token,
            batched=False,
            #desc="Running tokenizer on dataset",
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