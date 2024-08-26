from datasets import load_dataset


actions = {
    "walk": "I_WALK",
    "run": "I_RUN",
    "jump": "I_JUMP",
    "look": "I_LOOK"
    }

turns = {
    "around": "yyyy",
    "opposite": "yy"
}

directions = {
    "right": "I_TURN_RIGHT",
    "left": "I_TURN_LEFT"
}

nums = {
    "twice": "xx",
    "thrice": "xxx"
}

# all mappings together 
classes = [actions, turns, directions, nums]
# inverted turns and directions mappings
turns_inv = {v: k for k, v in turns.items()}
nums_inv = {v: k for k, v in nums.items()}

# variable binding function
# can be separated into at least 2 steps :
# 1. mapping turns
# 2. mapping nums

## specify assumptions ##
def variable_binding(l):

    # check if no nums or turns
    # return merged string
    if not any(x in l for x in list(turns_inv.keys())+list(nums_inv.keys())):
        return [' '.join(l)]

    # check for turns
    for n, v in turns_inv.items():
        if n not in l:
            continue
        ind = l.index(n)
        op = l[ind]
        # for around, need to perform action with each right or left turn
        if v == 'around':
            # last item in list is the action
            # 'turn' is also an action but its fused with the right and left turn commands
            # in case of 'turn' no need to append action to command
            # turn modifies the item before it
            if l[-1] in set(actions.values()):
                dec_act = (l[ind-1]+' '+l[-1]+' ')*len(op)
            else:
                dec_act = (l[ind-1]+' ')*len(op)
            # strip to remove trailing whitespace
            dec_act = dec_act.strip()
            l[ind] = dec_act

        # for opposite,  need to complete 180 turn and then perform action
        else:
            if l[-1] in set(actions.values()):
                dec_act = (l[ind-1]+' ')*len(op) + l[-1]
            else:
                dec_act = (l[ind-1]+' ')*len(op)
            # strip to remove trailing whitespace
            dec_act = dec_act.strip()
            l[ind] = dec_act

        # remove the turn placeholder and optionally the action
        l.pop(ind-1)
        if l[-1] in set(actions.values()):
            l.pop(-1)

    # check for nums (TODO : separate out)       
    for n, v in nums_inv.items():
        if n not in l:
            continue
        ind = l.index(n)
        op = l[ind]
        # num modifies all the items after it
        dec_act = ((' '.join(l[ind+1:])+' ')*len(op)).strip()
        l[ind] = dec_act

    # the first item is the modified string
    return l[:1]


def causal_model(command):

    l0=command.split()
    l0 = list(filter(lambda a: a != 'turn', l0))

    # layer 1: get phrase divisions
    if 'and' in l0:
        # maintain order of command
        l11=l0[:l0.index("and")] 
        l12= l0[l0.index("and")+1:]
    elif 'after' in l0:
        # reverse order of command
        l11= l0[l0.index("after")+1:]
        l12=l0[:l0.index("after")] 
    else:
        l11=l0
        l12=[]

    #l1=[l11,l12]

    # layer 2: apply interpretation function depending on word class
    l21 = [classes[[w in c for c in classes].index(True)][w] for w in l11]
    l22 = [classes[[w in c for c in classes].index(True)][w] for w in l12]

    # reverse items in list for easier parsing
    l21 = list(reversed(l21))
    l22 = list(reversed(l22))

    #l2=[l21,l22]

    # layer 3: variable binding
    l31 = variable_binding(l21)
    l32 = variable_binding(l22)

    l3=[l31,l32]

    # layer 4: action sequence output
    sequence = ' '.join([item for sublist in l3 for item in sublist])
    
    return sequence.strip()


if __name__ == '__main__':

    scan_simple = load_dataset('scan', 'simple', trust_remote_code=True)
    scan_length = load_dataset('scan', 'length', trust_remote_code=True)

    simple_train = scan_simple['train']
    simple_test = scan_simple['test']
    length_train = scan_length['train']
    length_test = scan_length['test']

    data_splits = [simple_train, simple_test, length_train, length_test]
    total_len = sum([len(s) for s in data_splits])

    accuracy = 0
    for dataset in data_splits:
        for x in dataset:
            command = x['commands']
            label = x['actions']
            output = causal_model(command)
            if label == output: accuracy += 1
    print('accuracy on simple and length splits : {}'.format(accuracy/total_len))