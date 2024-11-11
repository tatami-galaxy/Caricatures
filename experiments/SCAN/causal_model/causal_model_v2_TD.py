import copy

from datasets import load_dataset

# Top down CAM
# 1. Resolve C: Split sentence based on and /after
# 2. Resolve S: Identify and interpret twice/thrice
# 3. Resolve V: Identify and interpret opposite/around
# 4. Resolve D: Identify and interpret left/right/turn left/turn right
# 5. Resolve U: Identify and interpret all verbs


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
        else: l2.append([l])


    # STEP 3: Resolve V: Interpret opposite/around and handle direction repeats
    l3 = []
    ar_opp_interp = {'around':['turn', 'action']*4, 'opposite':['turn', 'turn', 'action']}
    for l in l2:
        new_l = []
        # has nested lists
        for nl in l:
            # find around/opposite, None otherwise
            item = copy.copy(nl)
            intersect = list(set(ar_opp_interp.keys()).intersection(set(nl)))
            ar_opp = intersect[0] if len(intersect) > 0 else None
            if ar_opp is not None:
                # resolve around/opposite
                resl = ar_opp_interp[ar_opp]
                item[item.index(ar_opp)] = resl
            new_l.append(item)
        l3.append(new_l)


    # STEP 4: Resolve D: Identify and interpret directions
    l4 = []
    dircetions = {'left': 'I_TURN_LEFT', 'right': 'I_TURN_RIGHT'}
    for l in l3:
        new_l = []
        # has nested lists
        for nl in l:
            item = copy.copy(nl)
            # get index of resolved around/opposite else None
            # using the fact that resolved item is a list
            # TODO: what if no around/opposite?
            resl_ind = [isinstance(i, list) for i in item].index(True) if any(isinstance(i, list) for i in item) else None
            # replace turns with actions using the given direction
            if resl_ind is not None:
                # get direction
                dir = item[resl_ind + 1]
                # replace turn with turn action
                item = [dircetions[dir] if i=='turn' else i for i in item[resl_ind]]


            new_l.append(item)
        l4.append(new_l)

    
    # STEP 5: Resolve U: Identify and replace all verbs



    # Resolve D: Interpret directions (left/right/turn left/turn right)
    l4 = []
    for sub in l3:
        new_sub = []
        skip_next = False
        for i, item in enumerate(sub):
            if skip_next:
                skip_next = False
                continue
            if item == 'turn' and i + 1 < len(sub):
                if sub[i + 1] == 'right':
                    new_sub.append('RTURN')
                    skip_next = True  # Skip 'right'
                elif sub[i + 1] == 'left':
                    new_sub.append('LTURN')
                    skip_next = True  # Skip 'left'
            elif item == 'right':
                new_sub.append('RTURN')
            elif item == 'left':
                new_sub.append('LTURN')
            else:
                new_sub.append(item)
        l4.append(new_sub)


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

    command = 'look around right twice and turn opposite left twice'
    #command = 'turn left twice and jump'
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