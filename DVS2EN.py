import time
import csv
import numpy as np
import matplotlib as plt
from copy import *
import os
import yaml
import scipy.io as sio
from brian2 import *
from brian2tools import *
import pandas as pd

from scipy.sparse import rand
import random
from deap import base
from deap import creator
from deap import tools
import json
import argparse

import pickle

import brian2genn

# prefs.devices.genn.cuda_path = '/usr/local/cuda-10.0'
# prefs.devices.genn.path = '/home/le/Installation/packages/genn-3.2.0'
#
# # set_device('cpp_standalone', directory='STDP_standalone',build_on_run=False)
# set_device('cpp_standalone')

random.seed = 2020
numpy.random.RandomState(seed = 2020)
import yaml

# load parameters
with open('MBSNN_params.yaml' , 'rb') as f:
    params = yaml.safe_load(f)

##global

nb_pn = params['NB_Neuron']['PN']
nb_kc = params['NB_Neuron']['KC']
nb_en = params['NB_Neuron']['EN']

nb_pn2kc = params['Neuron2Neuron']['nb_pn2kc']


def pn2kc(nb_pn, nb_kc, nb_pn2kc):
    d = float(nb_pn2kc) / float(nb_pn)
    matrix = rand(nb_pn, nb_kc, density = d, format = "csr", dtype = bool, random_state = 2020)
    pn2kc = nonzero(matrix)
    return pn2kc


wm_pn2kc = pn2kc(nb_pn , nb_kc , nb_pn2kc)


class MB_LE(object):

    def __init__(self, dvs_input, pars, w_pn2kc=wm_pn2kc, w_kc2kc=0):
        self.input = dvs_input
        self.w_kc2kc = w_kc2kc


        self.pn_neuron_idx = range(0 , nb_pn , nb_pn / 5)
        self.kc_neuron_idx = range(0 , nb_kc , nb_kc / 10)

        ## params realted to inhibition synapse learning(generation)
        v_kc2kc_a = pars[0] * mV
        tauw = pars[1] * ms
        i_kc2en = pars[2] * mV / ms
        tau_i_en = pars[3] * ms
        v_th_sp_kc = pars[4] * mV

        v_in2pn = params['Neuron2Neuron']['v_in2pn'] * mV
        v_pn2kc = params['Neuron2Neuron']['v_pn2kc'] * mV

        ## PN Neuron Model
        v_r_pn = params['Neuron_model']['PN']['resting_potential'] * mV
        v_th_pn = params['Neuron_model']['PN']['threshold_potential'] * mV
        v_reset_pn = params['Neuron_model']['PN']['reset_potential'] * mV
        tau_v_pn = params['Neuron_model']['PN']['time_constant'] * ms
        v_th_sp_pn = params['Neuron_model']['PN']['threshold_voltage_increase'] * mV
        rt_r_pn = params['Neuron_model']['PN']['rest_firing_rate'] * Hz
        tau_vth_pn = params['Neuron_model']['PN']['time_constant_threshold_voltage'] * ms

        ## the time_constant of vth is calcualted based on the resting rate of PN firing

        eqs_pn = '''
        dv/dt = (v_r_pn - v) / tau_v_pn : volt
        dvth/dt = (v_r_pn - 1*mV - vth) / tau_vth_pn : volt
        '''
        reset_pn = '''
        v = v_reset_pn
        vth += v_th_sp_pn
        '''

        ## KC Neuron Model
        v_r_kc = params['Neuron_model']['KC']['resting_potential'] * mV
        v_th_kc = params['Neuron_model']['KC']['threshold_potential'] * mV
        v_reset_kc = params['Neuron_model']['KC']['reset_potential'] * mV
        tau_v_kc = params['Neuron_model']['KC']['time_constant'] * ms
        tau_vth_kc = params['Neuron_model']['KC']['time_constant_threshold_voltage'] * ms
        #v_th_sp_kc = params['Neuron_model']['KC']['threshold_voltage_increase'] * mV


        # eqs_kc = ''' dv/dt = (v_r_kc - v) / tau_v_kc : volt
        #              dvth/dt = (v_r_kc - 1*mV - vth) / tau_vth_kc : volt
        #
        # '''
        # reset_kc = '''v = v_reset_kc
        # vth += v_th_sp_kc'''
        eqs_kc = ''' dv/dt = (v_r_kc - v) / tau_v_kc : volt

        '''
        reset_kc = '''v = v_reset_kc'''



        ## EN Neuron Model  ## EN model is LIF neuron model, using Isyn to set ESEP
        tau_i_en = tau_i_en
        v_ap_en = params['Neuron_model']['EN']['action_potential'] * mV
        v_r_en = v_r_pn
        tau_v_en = tau_v_pn
        v_reset_en = v_reset_pn

        eqs_en = '''
        dv/dt = (v_r_en - v)/tau_v_en + I : volt
        dI/dt = -I / tau_i_en    :  volt/second
        '''
        reset_en = '''
        v = v_reset_en
        I = 0 *mV/ms

        '''

        ####### define neurons

        self.dvs = SpikeGeneratorGroup(nb_pn, self.input[0], self.input[1]*ms)



        self.kcs = NeuronGroup(nb_kc , eqs_kc , threshold = 'v>=v_th_kc' , reset = reset_kc , method = 'exact',
                               namespace = {'v_r_kc': v_r_kc , 'tau_v_kc': tau_v_kc , 'v_th_kc': v_th_kc ,
                                            'tau_vth_kc': tau_vth_kc ,
                                            'v_reset_kc': v_reset_kc , 'v_th_sp_kc': v_th_sp_kc})

        # self.kcs.v = v_r_kc + np.random.permutation(nb_kc) / float(nb_kc) * 20 * mV
        # self.kcs.vth = self.kcs.v + np.flip(np.random.permutation(nb_kc) / float(nb_kc) * 5 * mV)
        #self.kcs.v = np.arange(v_r_kc, 0*mV, (0*mV-v_r_kc)/nb_kc)[0:nb_kc]
        self.kcs.v = v_r_kc


        self.kcs_a = NeuronGroup(nb_kc , eqs_kc , threshold = 'v>=v_th_kc' , reset = reset_kc ,method = 'exact',
                                 namespace = {'v_r_kc': v_r_kc , 'tau_v_kc': tau_v_kc , 'v_th_kc': v_th_kc ,
                                              'tau_vth_kc': tau_vth_kc ,
                                              'v_reset_kc': v_reset_kc , 'v_th_sp_kc': v_th_sp_kc})
        # self.kcs_a.v = v_r_kc + np.random.permutation(nb_kc) / float(nb_kc) * 20 * mV
        # self.kcs_a.vth = self.kcs_a.v + np.flip(np.random.permutation(nb_kc) / float(nb_kc) * 5 * mV)
        #self.kcs_a.v =  np.arange(v_r_kc, 0*mV, (0*mV-v_r_kc)/nb_kc)[0:nb_kc]
        self.kcs_a.v = v_r_kc

        self.en = NeuronGroup(nb_en , eqs_en , threshold = 'v>=v_ap_en' , reset = reset_en ,method = 'exact',
                              namespace = {'v_r_en': v_r_en , 'tau_v_en': tau_v_en , 'tau_i_en': tau_i_en ,
                                           'v_ap_en': v_ap_en , 'v_reset_en': v_reset_en})
        self.en.v = v_reset_en
        self.en.I = 0 * mV / ms

        self.en_a = NeuronGroup(nb_en , eqs_en , threshold = 'v>=v_ap_en' , reset = reset_en ,method = 'exact',
                                namespace = {'v_r_en': v_r_en , 'tau_v_en': tau_v_en , 'tau_i_en': tau_i_en ,
                                             'v_ap_en': v_ap_en , 'v_reset_en': v_reset_en})
        self.en_a.v = v_reset_en
        self.en_a.I = 0 * mV / ms

        ###### define coonnections:


        # dvs to kc connection
        self.S_dvs2kc = Synapses(self.dvs , self.kcs , on_pre = 'v += v_pn2kc' , namespace = {'v_pn2kc': v_pn2kc})
        self.S_dvs2kc.connect(i = w_pn2kc[0] , j = w_pn2kc[1])

        # pn to kc_a connection
        self.S_dvs2kc_a = Synapses(self.dvs , self.kcs_a , on_pre = 'v += v_pn2kc' , namespace = {'v_pn2kc': v_pn2kc})
        self.S_dvs2kc_a.connect(i = w_pn2kc[0] , j = w_pn2kc[1])

        # kc to en(MBON) connection
        self.S_kc2en = Synapses(self.kcs , self.en , on_pre = 'I += i_kc2en' , namespace = {'i_kc2en': i_kc2en})
        self.S_kc2en.connect()

        # kc_a to en_a connection
        self.S_kc_a2en_a = Synapses(self.kcs_a , self.en_a , on_pre = 'I += i_kc2en' , namespace = {'i_kc2en': i_kc2en})
        self.S_kc_a2en_a.connect()

        # kc2kc connection
        self.S_kc2kc_learning = Synapses(self.kcs , self.kcs_a ,
                                         ''' w : 1
                                             dApost/dt = -Apost/tauw  : 1 (event-driven)
                                         ''',
                                         on_pre = '''Apost = 1''',
                                         on_post = ''' w += Apost ''' ,
                                         namespace = {'tauw': tauw} ,
                                         )

        self.S_kc2kc_learning.connect(condition = 'i!=j' , skip_if_invalid = True)
        self.S_kc2kc_learning.Apost = 1
        self.S_kc2kc_learning.w = 0
        self.kc2kc_STM = StateMonitor(self.S_kc2kc_learning , 'w' , record = self.kc_neuron_idx)
        ##Weight Monitors:record weight change continuously
        if sum(self.w_kc2kc) > 0:

            self.S_kc2kc_learned = Synapses(self.kcs , self.kcs_a , '''w : 1''' ,
                                            on_pre = '''v_post -=  w * v_kc2kc''',
                                            namespace = {'v_kc2kc': v_kc2kc_a})
            #self.w_kc2kc[self.w_kc2kc<1] = NaN


            self.S_kc2kc_learned.connect(condition = 'i!=j' , skip_if_invalid = True)

            self.S_kc2kc_learned.w = self.w_kc2kc
            self.kc2kc_STM = StateMonitor(self.S_kc2kc_learned , 'w' , record = self.kc_neuron_idx)




        # if sum(w_kc2kc) == 0:  ##learnning mode
        #     print 'MBSNN is learning'
        #     self.S_kc2kc_a = Synapses(self.kcs , self.kcs_a ,
        #                               ''' w : 1
        #                                 dApost/dt = ( 1 - Apost ) / tauw : 1 (event-driven)
        #                                ''' ,
        #                               on_pre = '''v_post -=  w * v_kc2kc
        #                                            Apost = 0''' ,
        #                               on_post = ''' w += clip( Apost, 0.0, 0.9)''' ,
        #                               namespace = {'v_kc2kc': v_kc2kc_a , 'tauw': tauw} ,
        #                               )
        #
        #     self.S_kc2kc_a.connect(condition = 'i!=j' , skip_if_invalid = True)
        #     self.S_kc2kc_a.Apost = 0
        #     self.S_kc2kc_a.w = 0
        #
        # else:  ##testing mode
        #     print 'MBSNN is testing'
        #     self.S_kc2kc_a = Synapses(self.kcs , self.kcs_a , '''w : 1''' , on_pre = '''v_post -=  2 * w * v_kc2kc ''' ,
        #                               namespace = {'v_kc2kc': v_kc2kc_a})
        #     self.S_kc2kc_a.connect(condition = 'i!=j' , skip_if_invalid = True)
        #     self.S_kc2kc_a.w = w_kc2kc


        ##State Monitors:record membrane potential continuously



        self.KC_STM = StateMonitor(self.kcs , 'v' , record = self.kc_neuron_idx)
#        self.KC_VTM = StateMonitor(self.kcs , 'vth' , record = self.kc_neuron_idx)
        self.EN_STM = StateMonitor(self.en , 'v' , record = True)

        self.KC_A_STM = StateMonitor(self.kcs_a , 'v' , record = self.kc_neuron_idx)
        self.EN_A_STM = StateMonitor(self.en_a , 'v' , record = True)



        ##Spike Monitors: record the spikes whtin a neuron layer
        self.PN_SPM = SpikeMonitor(self.dvs)
        self.KC_SPM = SpikeMonitor(self.kcs)
        self.KC_A_SPM = SpikeMonitor(self.kcs_a)
        #
        self.EN_SPM = SpikeMonitor(self.en)
        self.EN_A_SPM = SpikeMonitor(self.en_a)
        #
        ##Rate Monitor: record the population firing rate
        self.IN_RTM = PopulationRateMonitor(self.dvs)
        self.KC_RTM = PopulationRateMonitor(self.kcs)
        self.KC_A_RTM = PopulationRateMonitor(self.kcs_a)

        self.EN_RTM = PopulationRateMonitor(self.en)
        self.EN_A_RTM = PopulationRateMonitor(self.en_a)

        ## link the elements to be a network

        self.net = Network(self.dvs, self.kcs, self.en, self.S_dvs2kc,  self.S_kc2en)

        self.net.add(self.kcs_a, self.en_a, self.S_dvs2kc_a, self.S_kc_a2en_a)

        ## add monitors into the network
        MTs = [self.KC_SPM , self.KC_A_SPM , self.EN_STM , self.EN_SPM , self.EN_RTM , self.EN_A_RTM , self.EN_A_STM,
               self.EN_A_SPM]
        self.net.add(MTs)

        self.net.add(self.PN_SPM, self.KC_RTM , self.KC_STM , self.KC_A_RTM, self.KC_A_STM)

    def run_sim(self):
        if sum(self.w_kc2kc) == 0:
            #print 'MBSNN is learning'
            self.net.add(self.S_kc2kc_learning, self.kc2kc_STM)
            self.net.run(self.input[1][-1]*ms - self.input[1][0]*ms)

        else:
            #print 'MBSNN is testing'
            self.net.add(self.S_kc2kc_learned , self.kc2kc_STM)
            self.dvs.set_spikes(self.input[0] , self.input[1] * ms)
            self.net.run(self.input[1][-1]*ms - self.input[1][0]*ms)






