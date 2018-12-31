import numpy as np
import pickle
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder


'''
An initial multinomial HMM implementation using for emission probabilities 
the frequency of each move (combined version)
'''


def flatten(lst):
    out = []
    for l in lst:
        tmp = []
        for ttt in l:
            tmp.append(ttt[0]-1)
        out.append(tmp)
    return out


def transform(lst):
    out = []
    for l in lst:
        for ll in l:
            out.append([ll])
    return out


def create_obs(file):
    with open(file, "rb") as fp:
        data = pickle.load(fp)
    data = flatten(data)
    stats = [0]*36
    for d in data:
        for i in range(0,36):
            stats[i]+=d.count(i)
    stats = list(map(lambda x: x/sum(stats),stats))
    lengths = list(map(lambda x: len(x), data))
    obs = np.array(transform(data))
    return obs,lengths,stats


def normalize(t):
    s = sum(t)
    return list(map(lambda x:x/s,t))


def create_and_fit_hmm(obs, lengths, statss):
    model = hmm.MultinomialHMM(n_components=4)
    final_stats = []
    for stats in statss:
        if (stats.count(0.0)!=0):
            inds = [i for i,e in enumerate(stats) if e==0.0]
            for i in inds:
                stats[i] = 0.00001
        final_stats.append(normalize(stats))
    print("Emission Probabilities")
    for ind, f in enumerate(final_stats):
        print("State" + str(ind)+":",f)
    model.startprob_ = np.array([0.25, 0.25, 0.25, 0.25])
    model.transmat_ = np.array([
        [0.85, 0.05, 0.05, 0.05],
        [0.05, 0.85, 0.05, 0.05],
        [0.05, 0.05, 0.85, 0.05],
        [0.05, 0.05, 0.05, 0.85]])
    model.emissionprob_ = np.array(final_stats)
    #model.fit(obs, lengths) # no need for fitting since all the parameters are initialized by hand
    return model


'''
Both the training set and the test set are evaluated here
'''
obs1, lengths1, stats1 = create_obs("conceder_combined_obs.txt")
obs2, lengths2, stats2  = create_obs("hardheaded_combined_obs.txt")
obs3, lengths3, stats3 = create_obs("random_combined_obs.txt")
obs4, lengths4, stats4  = create_obs("tft_combined_obs.txt")
obs = [obs1]+[obs2]+[obs3]+[obs4]
lengths = [lengths1]+[lengths2]+[lengths3]+[lengths4]
stats = [stats1]+[stats2]+[stats3]+[stats4]

model = create_and_fit_hmm(obs, lengths, stats)

print("Testing on the training set")
pred = list(model.predict(obs1))
print(pred.count(0),pred.count(1),pred.count(2),pred.count(3))
pred = list(model.predict(obs2))
print(pred.count(0),pred.count(1),pred.count(2),pred.count(3))
pred = list(model.predict(obs3))
print(pred.count(0),pred.count(1),pred.count(2),pred.count(3))
pred = list(model.predict(obs4))
print(pred.count(0),pred.count(1),pred.count(2),pred.count(3))

print("Testing")
obs, lengths, stats = create_obs("test_combined_obs.txt")
start = 0
# there are 14 tests because there are 2 agents in each testcase
for ind, ln in enumerate(lengths):
    print("-------------- Test "+str(ind+1)+" --------------")
    pred = list(model.predict(obs[start:start+ln]))
    print("Conceder: ",pred.count(0))
    print("HardHeaded: ", pred.count(1))
    print("Random: ", pred.count(2))
    print("TFT: ", pred.count(3))
    start = start + ln


