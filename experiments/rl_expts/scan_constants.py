# causal model
# longest command is 9 words : https://arxiv.org/pdf/1711.00350
command_max_len = 9
dummy_token = "<empty>"

# command type maps
actions = {
    "walk": "I_WALK",
    "run": "I_RUN",
    "jump": "I_JUMP",
    "look": "I_LOOK",
    "turn": dummy_token,
    dummy_token: dummy_token,
    }

turns = {
    "around": "yyyy",
    "opposite": "yy",
    dummy_token: dummy_token
}

directions = {
    "right": "I_TURN_RIGHT",
    "left": "I_TURN_LEFT",
    dummy_token: dummy_token
}

nums = {
    "twice": "xx",
    "thrice": "xxx",
    dummy_token: dummy_token
}

conjs = ["and", "after", dummy_token]

# command structure
command_structure = {
    0: actions,
    1: turns,
    2: directions,
    3: nums,
    4: conjs,
    5: actions,
    6: turns,
    7: directions,
    8: nums,
}

# tokenizer special tokens
special_tokens_dict = {
    "pad_token": "<pad>",
    "sep_token": "<sep>",
}