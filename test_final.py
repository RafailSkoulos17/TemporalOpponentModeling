import itertools
import json
import os
from collections import OrderedDict

import numpy as np
import pickle
import sys


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
            u1_curr = find_util(prof, bid[agent], issues)
            u1_prev = find_util(prof, bids[ind][agent], issues)

            u2_curr = find_util(opp, bid[agent], issues)
            u2_prev = find_util(opp, bids[ind][agent], issues)

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
    Update the logs with each move's type and some statistics about them.
    :param filename: The file that contains the negotiation  logs we want to update
    """
    test_combined = []  # the lists to store both the combination of moves and each move separately
    test_naive = []
    test_naive_utils = []

    with open(filename, "r") as fin:
        data = json.load(fin, object_pairs_hook=OrderedDict)

    issues = data["issues"]
    bids = data["bids"]
    profile_1 = data["Utility1"]
    profile_2 = data["Utility2"]

    temp_agent1 = []
    temp_agent2 = []  # used in case that we have agents from the same class negotiating
    temp1 = []
    temp2 = []  # used in case that we have agents from the same class negotiating
    temp_util1 = []
    temp_util2 = []

    for ind, bid in enumerate(bids[1:]):  # find each move's type, except form the first one

        moves, diff, diff_prev = find_moves_and_utils(ind, issues, profile_1, profile_2, bids, bid)
        # update the dictionary with the type of the move
        # compute the frequency of each move type
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

        if moves[0] and moves[1]:
            move_pair_num = move_pair_mapping[(moves[0], moves[1])]
            temp_agent1.append(move_pair_num)  # should be imported in the dataset
            temp_agent2.append(move_pair_mapping[(moves[1], moves[0])])

    test_combined.append(temp_agent1)
    test_combined.append(temp_agent2)
    test_naive.append(temp1)
    test_naive.append(temp2)
    test_naive_utils.append(temp_util1)
    test_naive_utils.append(temp_util2)
    return test_combined, test_naive, test_naive_utils


# another function to change the formatting
def transform(lst):
    out = []
    for l in lst:
        for ll in l:
            out.append([ll])
    return out


# function used just for changing the format of the data
def flatten(lst):
    out = []
    for l in lst:
        tmp = []
        for ttt in l:
            tmp.append(ttt[0])
        out.append(tmp)
    return out


# function used to normalize the data so that they fit the gaussian standards (-1,1)
def normalize(data):
    normalized = []
    for d in data:
        temp = []
        for t in d:
            temp.append(t / 100)
        normalized.append(temp)
    return normalized


# the main function that produces the observations in the needed format accompanied by the length of each sequence
def create_obs(data):
    normalized = normalize(data)
    lengths = list(map(lambda x: len(x), data))
    obs = np.array(transform(normalized))
    return obs, lengths


input_file = sys.argv[1]
test_combined, test_naive, test_naive_utils = update_logs(input_file)

# agent1, agent2 = testing_data(input_file)
obs, lengths = create_obs(test_combined)

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

# testing
start = 0
agents = ['Agent1', 'Agent2']

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
