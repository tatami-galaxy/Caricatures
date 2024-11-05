from datasets import load_dataset

# Top down CAM
# 1. Resolve C: Split sentence based on and /after
# 2. Resolve S: Identify and interpret twice/thrice
# 3. Resolve V: Identify and interpret opposite/around
# 4. Resolve D: Identify and interpret left/right/turn left/turn right
# 5. Resolve U: Identify and interpret all verbs

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

        # for opposite, need to complete 180 turn and then perform action
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

    # Step 0: Split the command into lexical items (words)
    l0=command.split()
    # STEP 1. Resolve C: Split based on 'and' or 'after'
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
    # merge
    l1=[l11,l12]


    # STEP 2. Resolve S: Interpret twice/thrice for repetition as individual elements
    l2 = []
    nums = {'twice':2, 'thrice':3}
    # Does doing it sequentially make sense since AR model also computes sequentially?
    for l in l1:
        # find twice/thrice, None otherwise
        intersect = list(set(nums.keys()).intersection(set(l)))
        num = intersect[0] if len(intersect) > 0 else None
        # repeat preceeding elements twice/thrice
        # merge repeated elements into nested list
        if num is not None:
            l[:l.index(num)] = [l[:l.index(num)]]*nums[num]
            l.remove(num)
        l2.append(l)


    # STEP 3: Resolve V: Interpret opposite/around and handle direction repeats
    l3 = []
    turns = {'opposite':2, 'around':4}
    turn_actions = {'right': 'I_TURN_RIGHT', 'left': 'I_TURN_LEFT'}
    for l in l2:
        l31 = []
        # has nested lists
        for nl in l:
            # find around/opposite, None otherwise
            intersect = list(set(turns.keys()).intersection(set(nl)))
            turn = intersect[0] if len(intersect) > 0 else None
            # ['look', 'opposite', 'right']
            # ['turn', 'opposite', 'right']
            if turn == 'opposite':
                pass
            # look around right
            elif turn == 'around':
                nl.remove(turn)
                


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

    ## testing ##

    command = 'look around right twice and turn left thrice'
    causal_model(command)

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
    for dataset in data_splits:
        for x in dataset:
            command = x['commands']
            label = x['actions']
            output = causal_model(command)
            if label == output: accuracy += 1
    print('accuracy on simple and length splits : {}'.format(accuracy/total_len))