import json
import os
import shutil
from collections import OrderedDict


def find_util(profile, bid):
    """
    Finds the utility for a bid, given the preference profile
    :param profile: The preference profile of the agent
    :param bid: The bid that is examined
    :return: The utility of the agent for the given bid
    """
    bid = bid.split(",")
    utility = 0
    for index, issue in enumerate(issues):
        utility += profile[issue]["weight"]*profile[issue][bid[index]]
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

# create the directory for the updated training logs
result_dir = "training_logs_updated"
if os.path.isdir(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir)


for root, dirs, files in os.walk("training_logs", topdown=False):   # run for all the training logs
    for log in files:
        with open(os.path.join(root,log), "r") as fin:
            data = json.load(fin, object_pairs_hook=OrderedDict)
        issues = data["issues"]
        bids = data["bids"]
        profile_1 = data["Utility1"]
        profile_2 = data["Utility2"]

        kind_of_moves1 = {}
        kind_of_moves2 = {}
        num_of_moves1 = 0
        num_of_moves2 = 0

        for ind, bid in enumerate(bids[1:]):    # find each move's type, except form the first one
            moves = []
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
                    u1_curr = find_util(prof, bid[agent])
                    u1_prev = find_util(prof, bids[ind][agent])

                    u2_curr = find_util(opp, bid[agent])
                    u2_prev = find_util(opp, bids[ind][agent])

                    move = find_move_type(u1_curr, u1_prev, u2_curr, u2_prev)
                    moves.append(move)
                else:
                    moves.append(None)

            # update the dictionary with the type of the move
            # compute the frequency of each move type
            if moves[0]:
                num_of_moves1 += 1
                if moves[0] in kind_of_moves1:
                    kind_of_moves1[moves[0]] += 1
                else:
                    kind_of_moves1[moves[0]] = 1
                data["bids"][ind+1]["move1"] = moves[0]
            if moves[1]:
                num_of_moves2 += 1
                if moves[1] in kind_of_moves2:
                    kind_of_moves2[moves[1]] += 1
                else:
                    kind_of_moves2[moves[1]] = 1
                data["bids"][ind+1]["move2"] = moves[1]

        for move, times in kind_of_moves1.items():
            kind_of_moves1[move] = "{0:.2f}".format(float(kind_of_moves1[move]) / num_of_moves1)

        for move, times in kind_of_moves2.items():
            kind_of_moves2[move] = "{0:.2f}".format(float(kind_of_moves2[move]) / num_of_moves2)

        data["kind_of_moves1"] = kind_of_moves1
        data["kind_of_moves2"] = kind_of_moves2

        # save the updated training log
        with open(os.path.join(result_dir, "updated_" + log), "w") as fout:
            json.dump(data, fout, indent=4, separators=(',', ': '))
