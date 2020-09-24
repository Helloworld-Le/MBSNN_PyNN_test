
from DVS import *
from PN2EN import *

from deap import base
from deap import creator
from deap import tools
import json
import argparse

import pickle

import brian2genn
#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
sys.setrecursionlimit(20000)

import time
import csv
import numpy as np

from copy import *
import os
import yaml
import scipy.io as sio
from brian2 import *
import brian2tools
from brian2tools import *
import pandas as pd
from scipy.sparse import rand
import brian2
import random
from deap import base
from deap import creator
from deap import tools
import json
import argparse

import pickle

import brian2genn
import matplotlib.pyplot as plt
from DVS import *
from PN2EN import *
random.seed = 2020
numpy.random.RandomState(seed = 2020)

input_file = '/home/le/Python_Plus/PycharmProjects/MBSNN0227/data/dvs_downsample_without_filter/'
output_file = '/home/le/Videos/0219normallense/v_20_EA'


## import dvs data
data = sio.loadmat(input_file + 'NB6_2020-02-19-11-20-17.mat')
idx_dvs = data['idx'][0]
t_dvs = data['t'][0]

print len(t_dvs)

## find when the camera starts to move : density of events is larger than the time period when camrea is staying still (10>1)
find_start_index = np.bincount(t_dvs.astype(int))
moving = np.where(find_start_index == 7)[0]
start = int(moving[0])
end = int(moving[-1])
start_t_index = np.where(t_dvs == start)[0][0]
end_t_index = np.where(t_dvs == end)[-1][-1]
print start_t_index, end_t_index

idx_input = idx_dvs[start_t_index:end_t_index].copy()
t_input = t_dvs[start_t_index:end_t_index].copy() - t_dvs[start_t_index]

print t_input

def find_index_for_distance(d, t_array):
    v = 20.0/1000
    t_l = d / v
    t_h = (d + 1) / v
    index_d = np.median(np.where((t_array >= t_l) & (t_array <= t_h)))
    return int(index_d)

##### distance = 100cm    t = d/(20cm/s)
index_d_100 = find_index_for_distance(100, t_input)
index_d_150 = find_index_for_distance(150, t_input)
index_d_200 = find_index_for_distance(200, t_input)
index_d_250 = find_index_for_distance(250, t_input)
index_d_300 = find_index_for_distance(300, t_input)

### the whole route is devided into 3 parts: 1) 0-1.5m: pre-learning,  2) 1.5-2.0m: learning  2.0-4m: learning

##pre learning: index
index_learning_start = index_d_150
index_learning_end = index_d_200

t_value_learning_start = t_input[index_learning_start]
t_value_learning_end = t_input[index_learning_end]

idx_pre_learning = idx_input[0:index_learning_start].copy()
t_pre_learning = t_input[0:index_learning_start].copy()

idx_learn = idx_input[index_learning_start: index_learning_end].copy()
t_learn = t_input[index_learning_start: index_learning_end].copy()

idx_test = idx_input[index_learning_start: -1].copy()
t_test = t_input[index_learning_start: -1].copy()

learn_input = [[idx_pre_learning,t_pre_learning],[idx_learn,t_learn]]
test_input = [[idx_pre_learning,t_pre_learning],[idx_test,t_test]]


def draw_rec(b, h):
    rect = plt.Rectangle((t_value_learning_start,b), (t_value_learning_end-t_value_learning_start), h, facecolor = "red", alpha = 0.5)
    return rect


def plot_result(mb):
    ax0 = subplot(511)
    brian_plot(mb.PN_SPM)
    ax0.add_patch(draw_rec(0, nb_pn))
    title('PN_Sikes' , loc = 'right')

    ax1 = subplot(512)
    brian_plot(mb.KC_SPM)
    ax1.add_patch(draw_rec(0, nb_kc))
    title('KC_Spikes', loc= 'right')

    ax2 = subplot(513)
    brian_plot(mb.KC_A_SPM)
    ax2.add_patch(draw_rec(0, nb_kc))
    title('KC_A_Spikes', loc= 'right')

    ax3 = subplot(514)
    #brian_plot(mb.EN_STM)
    brian_plot(mb.EN_A_STM, color = 'grey', alpha = 0.8)
    ax3.add_patch(draw_rec(-40,20))
    title('EN_V' , loc = 'right')

    ax4 = subplot(515)
    brian_plot(mb.EN_SPM)
    brian_plot(mb.EN_A_SPM, color = 'grey', alpha = 0.8)
    ax4.add_patch(draw_rec(-0.5,1.5))
    title('EN_Spikes' , loc = 'right')

    # ax5 = subplot(615)
    # brian_plot(mb.EN_A_SPM)
    # ax5.add_patch(draw_rec(-1,2))
    # title('EN_A_Spikes' , loc = 'right')

    # ax6 = subplot(515)
    # brian_plot(mb.EN_RTM, color = 'blue')
    # brian_plot(mb.EN_A_RTM, color = 'orange')
    # ax6.add_patch(draw_rec(min(mb.EN_RTM.rate),max(mb.EN_RTM.rate)))
    # title('EN_Rate' , loc = 'right')
    show()


def plot_PN(mb):
    subplot(511)
    brian_plot(mb.IN_SPM)
    title('DVS_Spikes', loc = 'right')

    subplot(512)
    brian_plot(mb.PN_SPM)
    title('PN_Spikes', loc = 'right')

    subplot(513)
    brian_plot(mb.PN_RTM)
    title('PN_Rate', loc = 'right')

    subplot(514)
    brian_plot(mb.PN_STM)
    title('PN_V', loc = 'right')

    subplot(515)
    brian_plot(mb.PN_VTM)
    title('PN_Vt', loc = 'right')
    show()


def plot_KC(mb):
    subplot(511)
    brian_plot(mb.PN_SPM)
    title('PN_Spikes' , loc = 'right')

    subplot(512)
    brian_plot(mb.KC_SPM)
    title('KC_Spikes' , loc = 'right')

    subplot(513)
    brian_plot(mb.KC_RTM, label = 'KC')
    brian_plot(mb.KC_A_RTM, label = 'PN')
    title('KC_Rate' , loc = 'right')

    subplot(514)
    brian_plot(mb.KC_STM)
    title('KC_V' , loc = 'right')

    subplot(515)
    brian_plot(mb.KC_A_STM)
    title('KC_A_V' , loc = 'right')
    show()





creator.create("FitnessMax" , base.Fitness , weights = (-1.0,))
creator.create("Individual" , list , fitness = creator.FitnessMax , v_log = [])


def initial():
    return [random.randint(1 , 20) * 0.5 , random.randint(1 , 10) , random.randint(1 , 10) * (0.1) , random.randint(1, 25) , random.randint(1 , 30) ]
    ## give the 1th generation by random generator or used previously evoloved results

# vkc2kc = pars[0]
# t_learning_range = pars[1]
# i_en2kc = pars[2]
# tau_i_en = pars[3]
# v_kc_th_increase = pars[4]



toolbox = base.Toolbox()

#                      Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool" , initial)

#                         Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')


toolbox.register("individual" , tools.initIterate , creator.Individual , toolbox.attr_bool)

# define the population to be a list of individuals
toolbox.register("population" , tools.initRepeat , list , toolbox.individual)


# the goal ('fitness') function to be maximized
def evalOneMax(individual):
    ## run the learning , with pre_learning to get the initial state, and learnning for
    mb_learn = MB_LE(dvs_input = learn_input, pars = individual)
    mb_learn.run_sim()
    # plot_PN(mb_learn)
    # plot_KC(mb_learn)
    # plot_result(mb_learn)
    # brian_plot(mb_learn.kc2kc_STM)
    # show()


    mb_test = MB_LE(dvs_input = test_input, pars = individual, w_kc2kc= mb_learn.S_kc2kc_learning.w)
    mb_test.run_sim()

    output_list0 = np.where((mb_test.EN_A_SPM.t / ms >= 2500) & (mb_test.EN_A_SPM.t / ms <= 7500))
    output_list1 = np.where((mb_test.EN_A_SPM.t / ms >= 7500) & (mb_test.EN_A_SPM.t / ms <= 10000))
    output_list2 = np.where((mb_test.EN_A_SPM.t / ms >= 10000) & (mb_test.EN_A_SPM.t / ms <= 15000))
    f = (len(output_list0[0])-500)**2 + len(output_list1[0])**2 + (len(output_list2[0])-500)**2

    individual.v_log = [len(output_list0[0]),len(output_list1[0]),len(output_list2[0])]
    return f,


# ----------
# Operator registration
# ----------
# register the goal / fitness function
toolbox.register("evaluate" , evalOneMax)

# register the crossover operator
toolbox.register("mate" , tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate" , tools.mutGaussian , indpb = 0.1)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select" , tools.selTournament , tournsize = 3)


# ----------

def main():

    # nb_individual = 2
    # genes = 5
    # pop = np.zeros((nb_individual,genes))
    # for i in range(0,2,1):
    #     var = [random.randint(1,10)*2, random.randint(1,20)*0.5, random.randint(1,20)*5, (random.randint(1,10))/10.0, random.randint(1,10)*4]
    #     pop[i][:] += var

    # pop.tolist()
    # create an initial population of 50 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n = 50)
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB , MUTPB = 0.5 , 0.5

    print("Start of evolution") , 'at:' , time.asctime(time.localtime(time.time()))

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate , pop))
    for ind , fit in zip(pop , fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    EA_log = []
    # Begin the evolution
    while g < 50:
        # A new generation
        population = {}
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop , len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone , offspring))

        # Apply crossover and mutation on the offspring
        for child1 , child2 in zip(offspring[::2] , offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1 , child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant , 2 , 1)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate , invalid_ind)
        for ind , fit in zip(invalid_ind , fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        Best_ind = tools.selBest(offspring , 1)[0]
        print("Best individual is %s, %s" % (Best_ind , Best_ind.fitness.values))

        EA_log.append([g , Best_ind , [Best_ind.fitness.values] , Best_ind.v_log])

        file = open(output_file + '/EA_log' + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')) + '.json' , 'w')
        json.dump(EA_log , file)
        print("-- Generation %i --" % g) , 'is loaded'


    print("-- End of (successful) evolution --")

    print time.asctime(time.localtime(time.time()))
    best_ind = tools.selBest(pop , 1)[0]
    print("Best individual is %s, %s" % (best_ind , best_ind.fitness.values))


main()
