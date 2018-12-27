import itertools
import os
import json
from collections import OrderedDict
import pickle


def training_data(type):
    for root, dirs, files in os.walk("training_logs_updated"):
        train_combined = []  # the lists to store both the combination of moves and each move separately
        train_naive = []
        for filename in files:
            temp_agent1 = []
            temp_agent2 = []    # used in case that we have agents from the same class negotiating
            temp1 = []
            temp2 = []  # used in case that we have agents from the same class negotiating
            if str(filename).count(type) == 2:  # if we 2 have agents from the wanted class
                with open(os.path.join(root, filename), "r") as fin:
                    data = json.load(fin, object_pairs_hook=OrderedDict)
                bids = data["bids"]
                for bid in bids:
                    if "move_pair_num" in bid:  # in that case both the "view" of agent1 and the "view" of agent2
                        temp_agent1.append([bid["move_pair_num"][1]])   # should be imported in the dataset
                        temp_agent2.append(
                            [move_pair_mapping[(bid["move_pair_num"][0][1], bid["move_pair_num"][0][0])]])
                    if "move1" in bid:
                        temp1.append([bid["move1"][1]])
                    if "move2" in bid:
                        temp2.append([bid["move2"][1]])
                train_combined.append(temp_agent1)
                train_combined.append(temp_agent2)
                train_naive.append(temp1)
                train_naive.append(temp2)
            elif str(filename).count(type) == 1:    # if only one of the 2 agents is the wanted
                with open(os.path.join(root, filename), "r") as fin:
                    data = json.load(fin, object_pairs_hook=OrderedDict)
                bids = data["bids"]
                for bid in bids:
                    if str(filename).find(type) < 10:   # if the wanted agent is agent1
                        if "move_pair_num" in bid:
                            temp_agent1.append([bid["move_pair_num"][1]])
                        if "move1" in bid:
                            temp1.append([bid["move1"][1]])
                    else:                               # if the wanted agent is agent2
                        if "move_pair_num" in bid:
                            temp_agent1.append(
                                [move_pair_mapping[(bid["move_pair_num"][0][1], bid["move_pair_num"][0][0])]])
                        if "move2" in bid:
                            temp1.append([bid["move2"][1]])
                train_combined.append(temp_agent1)
                train_naive.append(temp1)
        # and we save the two lists in a properly named txt file
        with open(type+"_combined_obs.txt", "wb") as fp:
            pickle.dump(train_combined,fp)
        with open(type+"_naive_obs.txt", "wb") as fp:
            pickle.dump(train_naive, fp)


def testing_data():
    for root, dirs, files in os.walk("test_logs_updated"):
        test_combined = []
        test_naive = []
        for filename in files:
            temp_agent1 = []
            temp_agent2 = []
            temp1 = []
            temp2 = []
            with open(os.path.join(root, filename), "r") as fin:
                data = json.load(fin, object_pairs_hook=OrderedDict)
            bids = data["bids"]
            for bid in bids:                # in the test data both "views" should be stored in the dataset
                if "move_pair_num" in bid:
                    temp_agent1.append([bid["move_pair_num"][1]])
                    temp_agent2.append(
                        [move_pair_mapping[(bid["move_pair_num"][0][1], bid["move_pair_num"][0][0])]])
                if "move1" in bid:
                    temp1.append([bid["move1"][1]])
                if "move2" in bid:
                    temp2.append([bid["move2"][1]])
            test_combined.append(temp_agent1)
            test_combined.append(temp_agent2)
            test_naive.append(temp1)
            test_naive.append(temp2)
        # and we save the two lists in a properly named txt file
        with open("test_combined_obs.txt", "wb") as fp:
            pickle.dump(test_combined,fp)
        with open("test_naive_obs.txt", "wb") as fp:
            pickle.dump(test_naive, fp)


moves_product = list(
    itertools.product(["nice", "silent", "fortunate", "unfortunate", "selfish", "concession"], repeat=2))
move_pair_mapping = dict([(y, x + 1) for x, y in enumerate(sorted(set(moves_product)))])

# in this section we create seperate datasets for each class
training_data("conceder")
training_data("hardheaded")
training_data("random")
training_data("tft")

# we also apply the same format on our test set
testing_data()

'''
# just a way to load the lists and see their content
with open("conceder_naive_obs.txt", "rb") as fp:
    t = pickle.load(fp)
for tt in t:
    print(tt)
print(len(t))
'''
