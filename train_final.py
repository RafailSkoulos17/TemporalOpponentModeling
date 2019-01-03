import itertools
import json
import os
from collections import OrderedDict
import numpy as np
import pickle
from hmmlearn import hmm


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


# The mapping of moves and pair of moves to numbers
move_mapping = {"concession": 1, "fortunate": 2, "nice": 3, "selfish": 4, "silent": 5, "unfortunate": 6}

moves_product = list(
    itertools.product(["nice", "silent", "fortunate", "unfortunate", "selfish", "concession"], repeat=2))
move_pair_mapping = dict([(y, x + 1) for x, y in enumerate(sorted(set(moves_product)))])


def find_moves_and_utils(ind, issues, profile_1, profile_2, bids, bid):
    """
        Finds the moves made in each bid
        :param ind: The index of the examined bid in the list of bids of the negotiation
        :param issues: The issues in the current negotiation
        :param profile_1: The profile of agent 1
        :param profile_2: The profile of agent 2
        :param bids: The bids of the negotiation
        :param bid: The currently examined bid
        :return: The moves of the two agents, and the difference of the utilities of the bids made in the current round
                 If its the second round also diff_prev is returned with the utilities of the first round. These
                 utilities were examined as features at the beginning but the results were not so promising.
        """
    moves = []
    diff = []
    diff_prev = []
    for agent in ["agent1", "agent2"]:
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


def update_logs(d, type):
    """
    Produces the needed features. In the current implementation it creates the list of combined moves for each type
    of agent (conceder, hardheaded, random, tft)
    :param d: The directory of the log files
    :param type: The type of agents for which the training set will be created
    :return: The training set for the wanted type of both the combined moves and just the simple moves. Also it returns
             the utilities of the bids for the needed class. In the current implementation only the combined moves
             are used as observations
    """

    for root, dirs, files in os.walk(d, topdown=False):  # run for all the training logs
        train_combined = []  # the lists to store both the combination of moves and each move separately
        train_naive = []
        train_naive_utils = []
        for log in files:
            if str(log).count(type) == 2: # if both agents are of the wanted type
                with open(os.path.join(root, log), "r") as fin:
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

                train_combined.append(temp_agent1)
                train_combined.append(temp_agent2)
                train_naive.append(temp1)
                train_naive.append(temp2)
                train_naive_utils.append(temp_util1)
                train_naive_utils.append(temp_util2)
            elif str(log).count(type) == 1:  # if only one of the 2 agents is the wanted
                with open(os.path.join(root, log), "r") as fin:
                    data = json.load(fin, object_pairs_hook=OrderedDict)

                issues = data["issues"]
                bids = data["bids"]
                profile_1 = data["Utility1"]
                profile_2 = data["Utility2"]
                temp_agent1 = []
                temp1 = []
                temp_util1 = []

                for ind, bid in enumerate(bids[1:]):  # find each move's type, except form the first one

                    moves, diff, diff_prev = find_moves_and_utils(ind, issues, profile_1, profile_2, bids, bid)
                    # update the dictionary with the type of the move
                    # compute the frequency of each move type
                    if str(log).find(type) < 10:  # if the wanted agent is agent1
                        if moves[0]:
                            move_num = move_mapping[moves[0]]
                            temp1.append(move_num)
                            if len(diff_prev) != 0:
                                temp_util1.append("{0:.3f}".format(diff_prev[0]))
                            temp_util1.append("{0:.3f}".format(diff[0]))

                        if moves[0] and moves[1]:
                            move_pair_num = move_pair_mapping[(moves[0], moves[1])]
                            temp_agent1.append(move_pair_num)  # should be imported in the dataset
                    else:
                        if moves[1]:
                            move_num = move_mapping[moves[1]]
                            temp1.append(move_num)
                            if len(diff_prev) != 0:
                                temp_util1.append("{0:.3f}".format(diff_prev[1]))
                            temp_util1.append("{0:.3f}".format(diff[1]))

                        if moves[0] and moves[1]:
                            temp_agent1.append(move_pair_mapping[(moves[1], moves[0])])

                train_combined.append(temp_agent1)
                train_naive.append(temp1)
                train_naive_utils.append(temp_util1)
    return train_combined, train_naive, train_naive_utils


# function used to normalize the data so that they fit the Gaussian standards (-1,1)
def normalize(data):
    normalized = []
    for d in data:
        temp = []
        for t in d:
            temp.append(t / 100)
        normalized.append(temp)
    return normalized


# Function to change the formatting of the data
def transform(lst):
    out = []
    for l in lst:
        for ll in l:
            out.append([ll])
    return out


# The main function that produces the observations in the needed format accompanied by the length of each sequence
def create_obs(data):
    normalized = normalize(data)
    lengths = list(map(lambda x: len(x), data))
    obs = np.array(transform(normalized))
    return obs,lengths


# Function used to train the hmm - only the prior probabilities are initialized
def create_and_fit_hmm(obs, lengths):
    model = hmm.GaussianHMM(n_components=2)
    model.startprob_ = np.array([0.5, 0.5])
    model.fit(obs,lengths)
    return model


# Function used to apply a type of leave-one-out cross validation to each category so that a better estimation
# of the counts of each class to be derived
def my_cross_val(obs, lengths):
    start = 0
    count0 = 0
    count1 = 0
    for ind,ln in enumerate(lengths):
        t_lengths = lengths.copy()
        t_lengths.pop(ind)
        model = create_and_fit_hmm(np.concatenate([obs[0:start],obs[start+ln:]]),t_lengths)
        res = model.predict(obs[start:start+ln])
        count0 += list(res).count(0)
        count1 += list(res).count(1)
        start += ln
    return count0, count1


'''
The main part of the calculations start here.
For each model the observations and the lengths are calculated 
Then each model is trained and the fitted counts are produced
After that also the "validated" counts are produced. We guarantee the validity of these counts by checking 
their relative differences from those of the fitted counts. 
'''

# conceder model
# train_naive and train_naive_utils are not eventually used since they were just some other possible features for use
train_combined, train_naive, train_naive_utils = update_logs("training_logs", "conceder")
obs1, lengths1 = create_obs(train_combined)
conceder_model = create_and_fit_hmm(obs1, lengths1)
pred = list(conceder_model.predict(obs1))
#print("Conceder fitted:",pred.count(0),pred.count(1))
conceder_diff = (pred.count(0)-pred.count(1))/(pred.count(0)+pred.count(1)) # calculation of the fitted relative difference

# hardheaded
train_combined, train_naive, train_naive_utils = update_logs("training_logs", "hardheaded")
obs2, lengths2 = create_obs(train_combined)
hardheaded_model = create_and_fit_hmm(obs2, lengths2)
pred = list(hardheaded_model.predict(obs2))
#print("HardHeaded fitted:",pred.count(0),pred.count(1))
hardheaded_diff = (pred.count(0)-pred.count(1))/(pred.count(0)+pred.count(1)) # calculation of the fitted relative difference

# random model
train_combined, train_naive, train_naive_utils = update_logs("training_logs", "random")
obs3, lengths3 = create_obs(train_combined)
random_model = create_and_fit_hmm(obs3, lengths3)
pred = list(random_model.predict(obs3))
#print("Random fitted:",pred.count(0),pred.count(1))
random_diff = (pred.count(0)-pred.count(1))/(pred.count(0)+pred.count(1)) # calculation of the fitted relative difference

# tft model
train_combined, train_naive, train_naive_utils = update_logs("training_logs", "tft")
obs4, lengths4 = create_obs(train_combined)
tft_model = create_and_fit_hmm(obs4, lengths4)
pred = list(tft_model.predict(obs4))
#print("TFT fitted:",pred.count(0),pred.count(1))
tft_diff = (pred.count(0)-pred.count(1))/(pred.count(0)+pred.count(1)) # calculation of the fitted relative difference

#print("---------------------------------------------")

# validating section
flag = False # flag to show that the validation "converged"
counts = 0 # a value used to ensure to loosen the criterion of stopping the validation procedure
threshold = 0.05 # the initial threshold of the difference between the validated and the fitted data
best = 100 # a value used just to ensure the termination of the procedure with the most decently converged value
while not flag:
    if counts > 5: # increase the threshold after 6 attempts are made
        counts = 0
        threshold += 0.01
    count0, count1 = my_cross_val(obs1, lengths1)
    conceder_diff_val = (count0 - count1)/(count0+count1) # calculation of the validated relative difference
    if abs(conceder_diff_val - conceder_diff) < best: # keep the best one so far just in case the procedure takes too long
        best = abs(conceder_diff_val - conceder_diff)
        best_val = conceder_diff_val
    if abs(conceder_diff_val-conceder_diff) < threshold:
        flag = True
    counts += 1
    if threshold > 0.1 and not flag: # if the procedure takes too long give the best value so far
        flag = True
        conceder_diff_val = best_val
#print("Conceder Validation:", count0, count1)
flag = False
counts = 0
threshold = 0.05
best = 100
while not flag:
    if counts > 5: # increase the threshold after 6 attempts are made
        counts = 0
        threshold += 0.01
    count0, count1 = my_cross_val(obs2, lengths2)
    hardheaded_diff_val = (count0 - count1)/(count0+count1) # calculation of the validated relative difference
    if abs(hardheaded_diff_val - hardheaded_diff) < best: # keep the best one so far just in case the procedure takes too long
        best = abs(hardheaded_diff_val - hardheaded_diff)
        best_val = hardheaded_diff_val
    if abs(hardheaded_diff_val - hardheaded_diff) < threshold:
        flag = True
    counts += 1
    if threshold > 0.1 and not flag: # if the procedure takes too long give the best value so far
        flag = True
        hardheaded_diff_val = best_val
#print("HardHeaded Validation:", count0, count1)
flag = False
counts = 0
threshold = 0.05
best = 100
while not flag:
    if counts > 5: # increase the threshold after 6 attempts are made
        counts = 0
        threshold += 0.01
    count0, count1 = my_cross_val(obs3, lengths3)
    random_diff_val = (count0 - count1)/(count0+count1) # calculation of the validated relative difference
    if abs(random_diff_val - random_diff) < best: # keep the best one so far just in case the procedure takes too long
        best = abs(random_diff_val - random_diff)
        best_val = random_diff_val
    if abs(random_diff_val - random_diff) < threshold:
        flag = True
    counts += 1
    if threshold > 0.1 and not flag: # if the procedure takes too long give the best value so far
        flag = True
        random_diff_val = best_val
#print("Random Validation:", count0, count1)
flag = False
counts = 0
threshold = 0.05
best = 100
while not flag:
    if counts > 5: # increase the threshold after 6 attempts are made
        counts = 0
        threshold += 0.01
    count0, count1 = my_cross_val(obs4, lengths4)
    tft_diff_val = (count0 - count1)/(count0+count1) # calculation of the validated relative difference
    if abs(tft_diff_val - tft_diff) < best: # keep the best one so far just in case the procedure takes too long
        best = abs(tft_diff_val - tft_diff)
        best_val = tft_diff_val
    if abs(tft_diff_val - tft_diff) < threshold:
        flag = True
    counts += 1
    if threshold > 0.1 and not flag: # if the procedure takes too long give the best value so far
        flag = True
        tft_diff_val = best_val

#print("TFT Validation:", count0, count1)

# finally the model and the produced differences are saved to be used for testing
to_save = [conceder_model, hardheaded_model,
           random_model, tft_model,
           conceder_diff, hardheaded_diff,
           random_diff, tft_diff,
           conceder_diff_val, hardheaded_diff_val,
           random_diff_val, tft_diff_val]
with open("models_metrics.txt", "wb") as fp:
    pickle.dump(to_save, fp)
