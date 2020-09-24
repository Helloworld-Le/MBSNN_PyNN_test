import argparse
import sys


import numpy as np

from PYNN_PN2EN import *
random.seed = 2020
numpy.random.RandomState(seed = 2020)


# vkc2kc = pars[0]
# t_learning_range = pars[1]
# i_en2kc = pars[2]
# tau_i_en = pars[3]
pars =[0.1, 0.5, 3, 3, 30]


def find_index_for_distance(d, t_array):
    v = 20.0/1000
    t_l = d / v
    t_h = (d + 1) / v
    index_d = np.min(np.where((t_array >= t_l) & (t_array <= t_h)))
    return int(index_d)


def get_spike_times(input):
    spike_time = []

    for i in range(nb_pn):
        index = np.where(input[0] == i)
        spike_time.append(np.ndarray.tolist(input[1][index]+1))
        #print min(np.ndarray.tolist(input[1][index]))

    print("number of spike soure neurons:", len(spike_time))
    return spike_time

def dvs2input(filename):

    ## import dvs data
    data = sio.loadmat(filename)
    idx_dvs = data['idx'][0]
    t_dvs = data['t'][0]

    ## find when the camera starts to move : density of events is larger than the time period when camrea is staying still (10>1)
    find_start_index = np.bincount(t_dvs.astype(int))
    moving = np.where(find_start_index == 7)[0]
    start = int(moving[0])
    end = int(moving[-1])
    start_t_index = np.where(t_dvs == start)[0][0]
    end_t_index = np.where(t_dvs == end)[-1][-1]

    idx_input = idx_dvs[start_t_index:end_t_index].copy()
    t_input = t_dvs[start_t_index:end_t_index].copy() - t_dvs[start_t_index] +1

    ##### distance = 100cm    t = d/(20cm/s)
    index_d_20 = find_index_for_distance(20,t_input)
    index_d_40 = find_index_for_distance(40,t_input)
    index_d_60 = find_index_for_distance(60 , t_input)
    index_d_80 = find_index_for_distance(80 , t_input)
    index_d_50 = find_index_for_distance(50 , t_input)
    index_d_100 = find_index_for_distance(100 , t_input)
    index_d_120 = find_index_for_distance(120 , t_input)
    index_d_150 = find_index_for_distance(150 , t_input)
    index_d_200 = find_index_for_distance(200 , t_input)
    index_d_300 = find_index_for_distance(300 , t_input)

    ### the whole route is devided into 3 parts: 1) 0-1.5m: pre-learning,  2) 1.5-2.0m: learning  2.0-4m: learning

    ##pre learning: index
    index_learning_start = 0
    index_learning_end = index_d_150

    t_value_learning_start = t_input[index_learning_start]
    t_value_learning_end = t_input[index_learning_end]

    idx_pre_learning = idx_input[0:index_learning_start].copy()
    t_pre_learning = t_input[0:index_learning_start].copy()

    idx_learn = idx_input[index_learning_start: index_learning_end].copy()
    t_learn = t_input[index_learning_start: index_learning_end].copy()

    idx_test = idx_input[index_learning_start: index_d_200].copy()
    t_test = t_input[index_learning_start: index_d_200].copy()

    t_flip = t_test.copy() * -1 + t_test[-1]
    t_flip = np.flipud(t_flip)
    idx_flip = np.flipud(idx_test.copy())

    idx_chop = np.concatenate((idx_input[index_d_100:index_d_150],idx_input[index_d_50:index_d_100],idx_input[index_learning_start:index_d_50],idx_input[index_d_150:index_d_200]))
    t_chop = np.concatenate((t_input[index_d_100:index_d_150]-t_input[index_d_100],t_input[index_d_50:index_d_100],t_input[index_learning_start:index_d_50]+t_input[index_d_100],t_input[index_d_150:index_d_200]))

    max_t = max(t_learn)
    learn_input = get_spike_times([idx_learn , t_learn])
    test_input = get_spike_times([idx_test , t_test])
    chop_input = get_spike_times([idx_chop , t_chop])
    flip_input = get_spike_times([idx_flip, t_flip])
    #
    #
    # learn_input = [idx_learn , t_learn]
    # test_input = [idx_input, t_input]



    return learn_input, test_input, chop_input, flip_input, max_t



learn_input, test_input, chop_input, flip_input, max_t = dvs2input('data/210PN_without_smooth/NB5_2020-02-19-11-19-03.mat')


mb_test = MB_LE(dvs_input = learn_input,sim_t = max_t)
mb_test.run_sim()
# mb_test1 = MB_LE(dvs_input = test_input,sim_t = max_t)
# mb_test1.run_sim()
# mb_test1 = MB_LE(dvs_input = test_input, pars = pars, w_kc2kc= mb_learn.S_kc2kc_learning.w)
# mb_test1.run_sim()
# plot_noise_EN(mb_test1, title = 'test sequence')
# #
# mb_test2 = MB_LE(dvs_input = chop_input, pars = pars, w_kc2kc= mb_learn.S_kc2kc_learning.w)
# mb_test2.run_sim()
# plot_noise_EN(mb_test2, title = 'distort sequence')
#
# mb_test3 = MB_LE(dvs_input = flip_input, pars = pars, w_kc2kc= mb_learn.S_kc2kc_learning.w)
# mb_test3.run_sim()
# plot_noise_EN(mb_test3, title = 'reverse sequence')
