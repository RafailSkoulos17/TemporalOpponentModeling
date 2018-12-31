import numpy as np
import pickle
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder

'''
This is the Gaussian version of HMM
It assigns a probability to each agent of the test set 
according to the difference of the counts and how close they 
are to the differences of the training set
Maybe it overfits a bit...
There is also a type of cross validation used for deriving 
the same type of results
Both kinds of results are printed at the end 
'''


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


# function previously used to find the greater dimension of the sequences in the training set - not used any more
def check_max():
    with open("conceder_combined_obs.txt", "rb") as fp:
        data = pickle.load(fp)
    data = flatten(data)
    conceder_l = max(list(map(lambda x: len(x), data)))
    max_l = conceder_l

    with open("hardheaded_combined_obs.txt", "rb") as fp:
        data = pickle.load(fp)
    data = flatten(data)
    hardheaded_l = max(list(map(lambda x: len(x), data)))
    if hardheaded_l > max_l:
        max_l = hardheaded_l

    with open("random_combined_obs.txt", "rb") as fp:
        data = pickle.load(fp)
    data = flatten(data)
    random_l = max(list(map(lambda x: len(x), data)))
    if random_l > max_l:
        max_l = random_l

    with open("tft_combined_obs.txt", "rb") as fp:
        data = pickle.load(fp)
    data = flatten(data)
    tft_l = max(list(map(lambda x: len(x), data)))
    if tft_l > max_l:
        max_l = tft_l

    with open("test_combined_obs.txt", "rb") as fp:
        data = pickle.load(fp)
    data = flatten(data)
    test_l = max(list(map(lambda x: len(x), data)))
    if test_l > max_l:
        max_l = test_l

    return max_l


# function used to append zeros at the end of lists with size lower than the wanted - not used anymore
def make_obs(normalized, l):
    obs = []
    for n in normalized:
        if len(n) < l:
            obs.append(n + [0.0] * (l - len(n)))
        else:
            obs.append(n)
    return obs


# another function to change the formatting
def transform(lst):
    out = []
    for l in lst:
        for ll in l:
            out.append([ll])
    return out


# the main function that produces the observations in the needed format accompanied by the length of each sequence
def create_obs(file):
    with open(file, "rb") as fp:
        data = pickle.load(fp)
    data = flatten(data)
    normalized = normalize(data)
    lengths = list(map(lambda x: len(x), data))
    obs = np.array(transform(normalized))
    #l = check_max()
    #obs = make_obs(normalized, l)
    return obs,lengths


# function used to train the hmm - only the prior probabilities are initialized
def create_and_fit_hmm(obs, lengths):
    model = hmm.GaussianHMM(n_components=2)
    model.startprob_ = np.array([0.5, 0.5])
    model.fit(obs,lengths)
    return model


# function used to apply a type of leave-one-out cross validation to each category so that a better estimation
# of the counts of each class to be derived (not an actual cross validation since the whole training set
# isn't used
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
        #print(start)
        start += ln
    return count0, count1


'''
The main part of the calculations start here.
For each model the observations and the lengths are calculated 
Then each model is trained and the fitted counts are produced
After that also the "cross validated" counts are produced
Finally the testing is performed by comparing the relative differences 
produced by the test cases to those produced by both the fitted sets 
and the cross validated ones.
'''

# conceder model
obs1, lengths1 = create_obs("conceder_combined_obs.txt")
#obs = np.array([np.array(xi) for xi in obs])
#print(normalized)
#obs = list(map(lambda x: list(LabelEncoder().fit_transform(x)), data))
conceder_model = create_and_fit_hmm(obs1, lengths1)
pred = list(conceder_model.predict(obs1))
print("Conceder fitted:",pred.count(0),pred.count(1))
conceder_diff = (pred.count(0)-pred.count(1))/(pred.count(0)+pred.count(1))

# hardheaded model
obs2, lengths2  = create_obs("hardheaded_combined_obs.txt")
hardheaded_model = create_and_fit_hmm(obs2, lengths2)
pred = list(hardheaded_model.predict(obs2))
print("HardHeaded fitted:",pred.count(0),pred.count(1))
hardheaded_diff = (pred.count(0)-pred.count(1))/(pred.count(0)+pred.count(1))

# random model
obs3, lengths3 = create_obs("random_combined_obs.txt")
random_model = create_and_fit_hmm(obs3, lengths3)
pred = list(random_model.predict(obs3))
print("Random fitted:",pred.count(0),pred.count(1))
random_diff = (pred.count(0)-pred.count(1))/(pred.count(0)+pred.count(1))

# tft model
obs4, lengths4  = create_obs("tft_combined_obs.txt")
tft_model = create_and_fit_hmm(obs4, lengths4)
pred = list(tft_model.predict(obs4))
print("TFT fitted:",pred.count(0),pred.count(1))
tft_diff = (pred.count(0)-pred.count(1))/(pred.count(0)+pred.count(1))

print("---------------------------------------------")

# validating
count0, count1 = my_cross_val(obs1,lengths1)
conceder_diff_val = (count0 - count1)/(count0+count1)
print("Conceder Validation:", count0, count1)
count0, count1 = my_cross_val(obs2,lengths2)
hardheaded_diff_val = (count0 - count1)/(count0+count1)
print("HardHeaded Validation:", count0, count1)
count0, count1 = my_cross_val(obs3,lengths3)
random_diff_val = (count0 - count1)/(count0+count1)
print("Random Validation:", count0, count1)
count0, count1 = my_cross_val(obs4,lengths4)
tft_diff_val = (count0 - count1)/(count0+count1)
print("TFT Validation:", count0, count1)

# testing
obs, lengths  = create_obs("test_combined_obs.txt")
#print(obs)
start = 0
# there are 14 tests because there are 2 agents in each testcase
for ind, ln in enumerate(lengths):
    print("-------------- Test "+str(ind+1)+" --------------")
    pred = list(conceder_model.predict(obs[start:start+ln]))
    diff1 = abs((pred.count(0)-pred.count(1))/(pred.count(0)+pred.count(1)) - conceder_diff)
    diff1_val = abs((pred.count(0) - pred.count(1)) / (pred.count(0) + pred.count(1)) - conceder_diff_val)
    print("Conceder: ",pred.count(0),pred.count(1))
    pred = list(hardheaded_model.predict(obs[start:start+ln]))
    diff2 = abs((pred.count(0)-pred.count(1))/(pred.count(0)+pred.count(1)) - hardheaded_diff)
    diff2_val = abs((pred.count(0) - pred.count(1)) / (pred.count(0) + pred.count(1)) - hardheaded_diff_val)
    print("Hardheaded: ", pred.count(0),pred.count(1))
    pred = list(random_model.predict(obs[start:start+ln]))
    diff3 = abs((pred.count(0)-pred.count(1))/(pred.count(0)+pred.count(1)) - random_diff)
    diff3_val = abs((pred.count(0) - pred.count(1)) / (pred.count(0) + pred.count(1)) - random_diff_val)
    print("Random: ",  pred.count(0),pred.count(1))
    pred = list(tft_model.predict(obs[start:start+ln]))
    diff4 = abs((pred.count(0)-pred.count(1))/(pred.count(0)+pred.count(1)) - tft_diff)
    diff4_val = abs((pred.count(0) - pred.count(1)) / (pred.count(0) + pred.count(1)) - tft_diff)
    print("TFT: ", pred.count(0),pred.count(1))
    #print(diff1, diff2, diff3, diff4)
    diff = diff1+diff2+diff3+diff4
    print("Conceder: ", 1 - diff1 / diff)
    print("Hardheaded: ",1 -  diff2 / diff)
    print("Random: ", 1 - diff3 / diff)
    print("TFT: ", 1 - diff4 / diff)
    diff = diff1_val + diff2_val + diff3_val + diff4_val
    print("From evaluation")
    print("Conceder: ", 1 - diff1_val  / diff)
    print("Hardheaded: ", 1 - diff2_val  / diff)
    print("Random: ", 1 - diff3_val  / diff)
    print("TFT: ", 1 - diff4_val  / diff)
    start = start + ln

