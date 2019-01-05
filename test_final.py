#!/usr/bin/env python3

import itertools
import json
import pickle
import sys
import numpy as np
from collections import OrderedDict


def find_util(profile, bid, issues):
    """
    Finds the utility for a bid, given the preference profile
    :param profile: The preference profile of the agent
    :param bid: The bid that is examined
    :param issues: The issues of the negotiation
    :return: The utility of the agent for the given bid
    """
    bid = bid.split(",")
    utility = 0
    for index, issue in enumerate(issues):
        utility += profile[issue]["weight"] * profile[issue][bid[index]]
    return utility


def find_move_type(u1_curr, u1_prev, u2_curr, u2_prev):
    """
    Finds the type of a move by comparing the change in the utility of the player (u1_curr - u1_prev)
    with the one for the opponent (u2_curr - u2_prev).
    The threshold used for nice and silent moves is 0.001
    :param u1_curr: Utility for the agent from the current bid
    :param u1_prev: Utility for the agent from the previous bid
    :param u2_curr: Utility for the opponent from the current bid
    :param u2_prev: Utility for the opponent from the previous bid
    :return: Type of move
    """
    move = "unknown"
    if -0.001 <= u1_curr - u1_prev <= 0.001 and u2_curr - u2_prev > 0:
        move = "nice"
    elif -0.001 <= u1_curr - u1_prev <= 0.001 and -0.001 <= u2_curr - u2_prev <= 0.001:
        move = "silent"
    elif u1_curr - u1_prev > 0 and u2_curr - u2_prev > 0:
        move = "fortunate"
    elif u1_curr - u1_prev <= 0 and u2_curr - u2_prev < 0:
        move = "unfortunate"
    elif u1_curr - u1_prev > 0 and u2_curr - u2_prev <= 0:
        move = "selfish"
    elif u1_curr - u1_prev < 0 and u2_curr - u2_prev >= 0:
        move = "concession"
    return move


# mapping of moves and pair of moves to numbers
move_mapping = {"concession": 1, "fortunate": 2, "nice": 3, "selfish": 4, "silent": 5, "unfortunate": 6}

moves_product = list(
    itertools.product(["nice", "silent", "fortunate", "unfortunate", "selfish", "concession"], repeat=2))
move_pair_mapping = dict([(y, x + 1) for x, y in enumerate(sorted(set(moves_product)))])


def find_moves_and_utils(ind, issues, profile_1, profile_2, bids, bid):
    """
    Finds the type of the moves for the 2 agents and the difference between
    the utility achieved for the 2 agents in the current round.If the round
    examined is the 2nd one(ind==0), then the difference
     between the utilities for the 1st round is also computed.
    :param ind: The index of the bids of the previous round.
    :param issues: The issues of the negotiation
    :param profile_1: The preference profile of the 1st agent
    :param profile_2: The preference profile of the 2nd agent
    :param bids: All the bids of the negotiation
    :param bid: The bid of the current round
    :return: The type of the moves for the 2 agents, the difference
     between the utilities for the 2 agents in the the current round.
     If the round examined is the 2nd one(ind==0), then the difference
     between the utilities for the 1st round is also returned.
    """
    moves = []
    diff = []
    diff_prev = []
    for agent in ["agent1", "agent2"]:  # run for each agent in  every round
        # in the round that an agent accepts an offer, the training log
        # has the following format:
        # "round": "143",
        # "agent1": "Apples,Milk,Almonds,None",
        # "accept": "agent2"
        # so we can't find the move's type
        if agent in bid:
            if agent == "agent1":
                prof = profile_1
                opp = profile_2
            else:
                prof = profile_2
                opp = profile_1
            u1_curr = find_util(prof, bid[agent], issues)   # current utility of the examined agent
            u1_prev = find_util(prof, bids[ind][agent], issues)     # previous utility of the examined agent

            u2_curr = find_util(opp, bid[agent], issues)    # current utility of the opponent agent
            u2_prev = find_util(opp, bids[ind][agent], issues)  # previous utility of the opponent agent

            diff.append(u1_curr - u2_curr)
            if ind == 0:
                diff_prev.append(u1_prev - u2_prev)

            move = find_move_type(u1_curr, u1_prev, u2_curr, u2_prev)
            moves.append(move)
        else:
            moves.append(None)
    return moves, diff, diff_prev


def update_logs(filename):
    """
    Examine a negotiation log and returns the move pairs for each round, the type of
    move for each agent in every round and the difference between the utilities for
    the 2 agents in every round.
    :param filename: The file that contains the negotiation logs we want to examine.
    :return: A list with the move pair for each round encoded, a list with the move
    of each agent in each round encoded, and a list with the difference between the
    utilities for the 2 agents in every round.
    """
    test_combined = []  # the list to store the combination of moves
    test_naive = []     # the list to store each move separately
    test_naive_utils = []   # the list to store the difference between the utility achieved for the 2 agents

    with open(filename, "r") as fin:
        data = json.load(fin, object_pairs_hook=OrderedDict)

    issues = data["issues"]
    bids = data["bids"]
    profile_1 = data["Utility1"]
    profile_2 = data["Utility2"]

    temp_agent1 = []
    temp_agent2 = []
    temp1 = []
    temp2 = []
    temp_util1 = []
    temp_util2 = []

    for ind, bid in enumerate(bids[1:]):  # find each move's type, except form the first one

        moves, diff, diff_prev = find_moves_and_utils(ind, issues, profile_1, profile_2, bids, bid)

        # update the list with the type of the move and the difference of the the utilities
        if moves[0]:
            move_num = move_mapping[moves[0]]
            temp1.append(move_num)
            if len(diff_prev) != 0:
                temp_util1.append("{0:.3f}".format(diff_prev[0]))
            temp_util1.append("{0:.3f}".format(diff[0]))

        if moves[1]:
            move_num = move_mapping[moves[1]]
            temp2.append(move_num)
            if len(diff_prev) != 0:
                temp_util2.append("{0:.3f}".format(diff_prev[1]))
            temp_util2.append("{0:.3f}".format(diff[1]))

        # update the list with the type of the pair of moves
        if moves[0] and moves[1]:
            move_pair_num = move_pair_mapping[(moves[0], moves[1])]
            temp_agent1.append(move_pair_num)
            temp_agent2.append(move_pair_mapping[(moves[1], moves[0])])

    test_combined.append(temp_agent1)
    test_combined.append(temp_agent2)
    test_naive.append(temp1)
    test_naive.append(temp2)
    test_naive_utils.append(temp_util1)
    test_naive_utils.append(temp_util2)
    return test_combined, test_naive, test_naive_utils


def transform(lst):
    """
    Takes as input a lists of lists and returns a list of lists where the inner lists j=ha sonly one element.
    :param lst: A list of lists.
    :return: A list of lists in which the inner lists contain only one element.
    """
    out = []
    for l in lst:
        for ll in l:
            out.append([ll])
    return out


# function used just for changing the format of the data
def flatten(lst):
    """
    Takes an input a list with multiple nested lists and returns a list
    of lists with the first element of each nested list.
    :param lst: A list with multiple nested lists.
    :return: A list of lists with the first element of each nested list
    """
    out = []
    for l in lst:
        tmp = []
        for ttt in l:
            tmp.append(ttt[0])
        out.append(tmp)
    return out


def normalize(data):
    """
    Normalize the data so that they fit the gaussian standards (-1,1)
    :param data: A list of lists
    :return: Normalized data
    """
    normalized = []
    for d in data:
        temp = []
        for t in d:
            temp.append(t / 100)
        normalized.append(temp)
    return normalized


def create_obs(data):
    """
    Produces the observations in the needed format accompanied by the length of each sequence
    :param data: A list of lists
    :return: Data normalized in the needed format and the length of each sequence.
    """
    normalized = normalize(data)
    lengths = list(map(lambda x: len(x), data))
    obs = np.array(transform(normalized))
    return obs, lengths


input_file = sys.argv[1]  # read the file given as input

# process the logs of the negotiation and get some outcomes
test_combined, test_naive, test_naive_utils = update_logs(input_file)

obs, lengths = create_obs(test_combined)  # compute the observations and their lengths

# load the trained model and its evaluation metrics
with open('models_metrics.txt', 'rb') as f:
    model_metrics = pickle.load(f)

conceder_model = model_metrics[0]
hardheaded_model = model_metrics[1]
random_model = model_metrics[2]
tft_model = model_metrics[3]
conceder_diff_val = model_metrics[8]
hardheaded_diff_val = model_metrics[9]
random_diff_val = model_metrics[10]
tft_diff_val = model_metrics[11]

# testing phase
start = 0
agents = ['Agent1', 'Agent2']

# print the outcome for each agent and each negotiation strategy
for ind, ln in enumerate(lengths):
    print("\n--------------  " + agents[ind] + "  --------------")

    pred = list(conceder_model.predict(obs[start:start + ln]))
    diff1_val = abs((pred.count(0) - pred.count(1)) / (pred.count(0) + pred.count(1)) - conceder_diff_val)

    pred = list(hardheaded_model.predict(obs[start:start + ln]))
    diff2_val = abs((pred.count(0) - pred.count(1)) / (pred.count(0) + pred.count(1)) - hardheaded_diff_val)

    pred = list(random_model.predict(obs[start:start + ln]))
    diff3_val = abs((pred.count(0) - pred.count(1)) / (pred.count(0) + pred.count(1)) - random_diff_val)

    pred = list(tft_model.predict(obs[start:start + ln]))
    diff4_val = abs((pred.count(0) - pred.count(1)) / (pred.count(0) + pred.count(1)) - tft_diff_val)

    diff = diff1_val + diff2_val + diff3_val + diff4_val
    print("Conceder: ", 1 - diff1_val / diff)
    print("Hardheaded: ", 1 - diff2_val / diff)
    print("Random: ", 1 - diff3_val / diff)
    print("TFT: ", 1 - diff4_val / diff)
    start = start + ln
