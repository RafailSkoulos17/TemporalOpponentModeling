import numpy as np
import pickle
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder

'''
An initial multinomial HMM implementation using for emission probabilities 
the frequency of each move (combined version)
'''

WEIGHT1 = 0.9
WEIGHT2 = 1.1
SPECIAL_WEIGHT1 = 0.7
SPECIAL_WEIGHT2 = 1.3
def flatten(lst):
    out = []
    for l in lst:
        tmp = []
        for ttt in l:
            tmp.append(ttt[0] - 1)
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
    stats = [0] * 36
    for d in data:
        for i in range(0, 36):
            stats[i] += d.count(i)
    stats = list(map(lambda x: x / sum(stats), stats))
    lengths = list(map(lambda x: len(x), data))
    obs = np.array(transform(data))
    return obs, lengths, stats


def normalize(t):
    s = sum(t)
    return list(map(lambda x: x / s, t))


def create_and_fit_multinomial_hmm(obs, lengths, statss):
    model = hmm.MultinomialHMM(n_components=4)
    final_stats = []
    for stats in statss:
        if (stats.count(0.0) != 0):
            inds = [i for i, e in enumerate(stats) if e == 0.0]
            for i in inds:
                stats[i] = 0.00001
        final_stats.append(normalize(stats))
    print("Emission Probabilities")
    for ind, f in enumerate(final_stats):
        print("State" + str(ind) + ":", f)
    model.startprob_ = np.array([0.25, 0.25, 0.25, 0.25])
    model.transmat_ = np.array([
        [0.85, 0.05, 0.05, 0.05],
        [0.05, 0.85, 0.05, 0.05],
        [0.05, 0.05, 0.85, 0.05],
        [0.05, 0.05, 0.05, 0.85]])
    model.emissionprob_ = np.array(final_stats)
    # model.fit(obs, lengths) # no need for fitting since all the parameters are initialized by hand
    return model


# function used to train the hmm - only the prior probabilities are initialized
def create_and_fit_hmm(obs, lengths):
    model = hmm.GaussianHMM(n_components=2)
    model.startprob_ = np.array([0.5, 0.5])
    model.fit(obs, lengths)
    return model


def predict(multinomial_model, model_list, obs, start, ln):
    pred1 = list(multinomial_model.predict(obs[start:start + ln]))
    count_all = pred1.count(0) + pred1.count(1) + pred1.count(2) + pred1.count(3)

    diff_m1 = abs((2 * pred1.count(0) - count_all) / count_all - diff_multi_conceder)
    diff_m2 = abs((2 * pred1.count(1) - count_all) / count_all - diff_multi_hardheaded)
    diff_m3 = abs((2 * pred1.count(2) - count_all) / count_all - diff_multi_random)
    diff_m4 = abs((2 * pred1.count(3) - count_all) / count_all - diff_multi_ttf)
    diff_mall = diff_m1 + diff_m2 + diff_m3 + diff_m4

    pred_conceder = list(model_list[0].predict(obs[start:start + ln]))
    diff1 = abs((pred_conceder.count(0) - pred_conceder.count(1)) / (
            pred_conceder.count(0) + pred_conceder.count(1)) - conceder_diff)

    pred_hardheaded = list(model_list[1].predict(obs[start:start + ln]))
    diff2 = abs((pred_hardheaded.count(0) - pred_hardheaded.count(1)) / (
            pred_hardheaded.count(0) + pred_hardheaded.count(1)) - hardheaded_diff)

    pred_random = list(model_list[2].predict(obs[start:start + ln]))
    diff3 = abs(
        (pred_random.count(0) - pred_random.count(1)) / (pred_random.count(0) + pred_random.count(1)) - random_diff)

    pred_ttf = list(model_list[3].predict(obs[start:start + ln]))
    diff4 = abs((pred_ttf.count(0) - pred_ttf.count(1)) / (pred_ttf.count(0) + pred_ttf.count(1)) - tft_diff)
    diff = diff1 + diff2 + diff3 + diff4

    print("Conceeder " + str(1 - diff_m1 / diff_mall) + "  " + str(1 - diff1 / diff))
    print("Hardheaded " + str(1 - diff_m2 / diff_mall) + "  " + str(1 - diff2 / diff))
    print("Random " + str(1 - diff_m3 / diff_mall) + "  " + str(1 - diff3 / diff))
    print("TTF " + str(1 - diff_m4 / diff_mall) + "  " + str(1 - diff4 / diff))

    conceder_prob = (WEIGHT1 * (1 - diff_m1 / diff_mall) + WEIGHT2 * (1 - diff1 / diff)) / 2
    hardheaded_prob = (WEIGHT1 * (1 - diff_m2 / diff_mall) + WEIGHT2 * (1 - diff2 / diff)) / 2

    # added some special weights as multinomial tends to choose frequently random
    random_prob = (SPECIAL_WEIGHT1 * (1 - diff_m3 / diff_mall) + SPECIAL_WEIGHT2 * (1 - diff3 / diff)) / 2
    ttf_prob = (WEIGHT1 * (1 - diff_m4 / diff_mall) + WEIGHT2 * (1 - diff4 / diff)) / 2
    print("------")
    print("Conceeder " + str(conceder_prob))
    print("Hardheaded " + str(hardheaded_prob))
    print("Random " + str(random_prob))
    print("TTF " + str(ttf_prob))


obs1, lengths1, stats1 = create_obs("conceder_combined_obs.txt")
obs2, lengths2, stats2 = create_obs("hardheaded_combined_obs.txt")
obs3, lengths3, stats3 = create_obs("random_combined_obs.txt")
obs4, lengths4, stats4 = create_obs("tft_combined_obs.txt")
obs = [obs1] + [obs2] + [obs3] + [obs4]
lengths = [lengths1] + [lengths2] + [lengths3] + [lengths4]
stats = [stats1] + [stats2] + [stats3] + [stats4]

model_multinomial = create_and_fit_multinomial_hmm(obs, lengths, stats)
pred = list(model_multinomial.predict(obs1))
count_all = pred.count(0) + pred.count(1) + pred.count(2) + pred.count(3)
diff_multi_conceder = abs((2 * pred.count(0) - count_all) / count_all)
diff_multi_hardheaded = abs((2 * pred.count(1) - count_all) / count_all)
diff_multi_random = abs((2 * pred.count(2) - count_all) / count_all)
diff_multi_ttf = abs((2 * pred.count(3) - count_all) / count_all)

print("Testing on the training set")
pred = list(model_multinomial.predict(obs1))
print(pred.count(0), pred.count(1), pred.count(2), pred.count(3))
pred = list(model_multinomial.predict(obs2))
print(pred.count(0), pred.count(1), pred.count(2), pred.count(3))
pred = list(model_multinomial.predict(obs3))
print(pred.count(0), pred.count(1), pred.count(2), pred.count(3))
pred = list(model_multinomial.predict(obs4))
print(pred.count(0), pred.count(1), pred.count(2), pred.count(3))

# conceder model
conceder_model = create_and_fit_hmm(obs1, lengths1)
pred = list(conceder_model.predict(obs1))
print("Conceder fitted:", pred.count(0), pred.count(1))
conceder_diff = (pred.count(0) - pred.count(1)) / (pred.count(0) + pred.count(1))

# hardheaded model
hardheaded_model = create_and_fit_hmm(obs2, lengths2)
pred = list(hardheaded_model.predict(obs2))
print("HardHeaded fitted:", pred.count(0), pred.count(1))
hardheaded_diff = (pred.count(0) - pred.count(1)) / (pred.count(0) + pred.count(1))

# random model
random_model = create_and_fit_hmm(obs3, lengths3)
pred = list(random_model.predict(obs3))
print("Random fitted:", pred.count(0), pred.count(1))
random_diff = (pred.count(0) - pred.count(1)) / (pred.count(0) + pred.count(1))

# tft model
tft_model = create_and_fit_hmm(obs4, lengths4)
pred = list(tft_model.predict(obs4))
print("TFT fitted:", pred.count(0), pred.count(1))
tft_diff = (pred.count(0) - pred.count(1)) / (pred.count(0) + pred.count(1))

print("---------------------------------------------")

print("Testing")
obs, lengths, stats = create_obs("test_combined_obs.txt")
start = 0
model_list = [conceder_model, hardheaded_model, random_model, tft_model]
# there are 14 tests because there are 2 agents in each testcase
for ind, ln in enumerate(lengths):
    print("-------------- Test " + str(ind + 1) + " --------------")
    predict(model_multinomial, model_list, obs, start, ln)
    start = start + ln
