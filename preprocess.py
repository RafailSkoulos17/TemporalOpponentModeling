import itertools
import json
import os
import shutil
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


def update_logs(d):
    """
    Update the logs with each move's type and some statistics about them.
    :param d: The directory of the log files
    """
    # create the directory for the updated logs
    new_dir = d + "_updated"
    if os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)

    for root, dirs, files in os.walk(d, topdown=False):  # run for all the training logs
        for log in files:
            with open(os.path.join(root, log), "r") as fin:
                data = json.load(fin, object_pairs_hook=OrderedDict)

            issues = data["issues"]
            bids = data["bids"]
            profile_1 = data["Utility1"]
            profile_2 = data["Utility2"]

            kind_of_moves1 = {}
            kind_of_moves2 = {}
            kind_of_moves_both = {}
            num_of_moves1 = 0
            num_of_moves2 = 0
            num_of_moves_both = 0

            for ind, bid in enumerate(bids[1:]):  # find each move's type, except form the first one
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

                        diff.append(u1_curr-u2_curr)
                        if ind == 0:
                            diff_prev.append(u1_prev-u2_prev)

                        move = find_move_type(u1_curr, u1_prev, u2_curr, u2_prev)
                        moves.append(move)
                    else:
                        moves.append(None)

                # update the dictionary with the type of the move
                # compute the frequency of each move type
                if moves[0]:
                    num_of_moves1 += 1
                    move_num = move_mapping[moves[0]]
                    if move_num in kind_of_moves1:
                        kind_of_moves1[move_num] += 1
                    else:
                        kind_of_moves1[move_num] = 1
                    data["bids"][ind + 1]["move1"] = [moves[0], move_num]
                    data["bids"][ind + 1]["diff1"] = "{0:.3f}".format(diff[0])
                    if len(diff_prev) != 0:
                        data["bids"][ind]["diff1"] = "{0:.3f}".format(diff_prev[0])
                if moves[1]:
                    num_of_moves2 += 1
                    move_num = move_mapping[moves[1]]
                    if move_num in kind_of_moves2:
                        kind_of_moves2[move_num] += 1
                    else:
                        kind_of_moves2[move_num] = 1
                    data["bids"][ind + 1]["move2"] = [moves[1], move_num]
                    data["bids"][ind + 1]["diff2"] = "{0:.3f}".format(diff[1])
                    if len(diff_prev) != 0:
                        data["bids"][ind]["diff2"] = "{0:.3f}".format(diff_prev[1])

                if moves[0] and moves[1]:
                    num_of_moves_both += 1
                    move_pair_num = move_pair_mapping[(moves[0], moves[1])]
                    data["bids"][ind + 1]["move_pair_num"] = [(moves[0], moves[1]), move_pair_num]
                    data["bids"][ind + 1]["combined_diff"] = "{0:.3f}".format(diff[0]-diff[1])
                    if len(diff_prev) != 0:
                        data["bids"][ind]["combined_diff"] = "{0:.3f}".format(diff_prev[0] - diff_prev[1])
                    if move_pair_num in kind_of_moves_both:
                        kind_of_moves_both[move_pair_num] += 1
                    else:
                        kind_of_moves_both[move_pair_num] = 1

            for move, times in kind_of_moves1.items():
                kind_of_moves1[move] = "{0:.2f}".format(float(kind_of_moves1[move]) / num_of_moves1)

            for move, times in kind_of_moves2.items():
                kind_of_moves2[move] = "{0:.2f}".format(float(kind_of_moves2[move]) / num_of_moves2)

            for move, times in kind_of_moves_both.items():
                kind_of_moves_both[move] = "{0:.2f}".format(float(kind_of_moves_both[move]) / num_of_moves_both)

            data["kind_of_moves1"] = kind_of_moves1
            data["kind_of_moves2"] = kind_of_moves2
            data["kind_of_moves_both"] = kind_of_moves_both

            # save the updated training log
            with open(os.path.join(new_dir, "updated_" + log), "w") as fout:
                json.dump(data, fout, indent=4, separators=(',', ': '))


# the directories of tha train and test logs
test_dir = "test_logs"
train_dir = "training_logs"

# call the functions that update the logs
update_logs(train_dir)
update_logs(test_dir)
