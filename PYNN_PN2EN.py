
import matplotlib as plt
from copy import *
import os
import pyyaml
import scipy.io as sio

from scipy.sparse import rand

import random

import json
import argparse

import pickle

import argparse

from numpy import arange
from pyNN.utility import get_simulator, init_logging, normalized_filename
import numpy as np

import pylab as plt
import spynnaker8 as sim
from pyNN.utility.plotting import Figure, Panel

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=14)
# +---------------------------------------------------------------------------+
# | General Parameters                                                        |
# +---------------------------------------------------------------------------+

# Population parameters
model = sim.IF_curr_exp
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 50)
cell_params = {'cm': 0.25,
               'i_offset': 0.0,
               'tau_m': 20.0,
               'tau_refrac': 2.0,
               'tau_syn_E': 5.0,
               'tau_syn_I': 5.0,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -50.0
               }
# === Configure the simulator ================================================

# sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.", {"action": "store_true"}),
#                              ("--debug", "Print debugging information"))

options = argparse.Namespace(debug=None, plot_figure=True, simulator='nest')


if options.debug:
    init_logging(None, debug=True)


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

    def __init__(self, dvs_input, sim_t, w_kc2kc=0):

        self.w_kc2kc = w_kc2kc

        self.sim_t = sim_t

        self.pn_neuron_idx = range(0 , nb_pn , nb_pn / 5)
        self.kc_neuron_idx = range(0 , nb_kc , nb_kc / 10)

        self.spike_source = sim.Population(nb_pn, sim.SpikeSourceArray(spike_times = dvs_input))
        self.pns = sim.Population(nb_pn , model(**cell_params), label = "PN")
        self.kcs = sim.Population(nb_kc , model(**cell_params) , label = "KC")
        self.kcs_a = sim.Population(nb_kc ,model(**cell_params), label = "KC_A")
        self.ens = sim.Population(nb_en ,model(**cell_params) , label = "EN")
        self.ens_a = sim.Population(nb_en , model(**cell_params), label = "EN_A")

        self.dvs2pn = sim.Projection(self.spike_source , self.pns , sim.OneToOneConnector() ,
                                    sim.StaticSynapse(weight = 0.1 , delay = 1.0) ,
                                    receptor_type = 'excitatory')
        self.pn2kc = sim.Projection(self.pns , self.kcs , sim.FixedTotalNumberConnector(nb_pn2kc*nb_kc) ,
                               sim.StaticSynapse(weight = 0.1 , delay = 1.0) ,
                               receptor_type = 'excitatory')
        self.pn2kc_a = sim.Projection(self.pns , self.kcs_a , sim.FixedTotalNumberConnector(nb_pn2kc*nb_kc) ,
                               sim.StaticSynapse(weight = 0.1 , delay = 1.0) ,
                               receptor_type = 'excitatory')
        self.kc2en = sim.Projection(self.kcs , self.ens , sim.AllToAllConnector() ,
                               sim.StaticSynapse(weight = 0.1 , delay = 1.0) ,
                               receptor_type = 'excitatory')
        self.kc_a2en_a = sim.Projection(self.kcs_a , self.ens_a , sim.AllToAllConnector() ,
                               sim.StaticSynapse(weight = 0.1 , delay = 1.0) ,
                               receptor_type = 'excitatory')

        self.MBONs = self.ens + self.ens_a
        self.MBONs.record(['v'])  # , 'u'])
    def run_sim(self):
        sim.run(self.sim_t)

        # === Save the results, optionally plot a figure =============================

        filename = normalized_filename("Results" , "Izhikevich" , "pkl" ,
                                       options.simulator , sim.num_processes())
        self.MBONs.write_data(filename , annotations = {'script_name': __file__})

        if options.plot_figure:
            from pyNN.utility.plotting import Figure , Panel
            figure_filename = filename.replace("pkl" , "png")
            data = self.MBONs.get_data().segments[0]
            v = data.filter(name = "v")[0]
            # u = data.filter(name="u")[0]
            Figure(
                Panel(v , ylabel = "Membrane potential (mV)" , xticks = True ,
                      xlabel = "Time (ms)" , yticks = True) ,
                # Panel(u, ylabel="u variable (units?)"),
                annotations = "Simulated with %s" % options.simulator.upper()
            ).save(figure_filename)
            print(figure_filename)

        # === Clean up and quit ========================================================

        sim.end()

        # else:
        #     print 'MBSNN is testing'
        #     self.net.add(self.S_kc2kc_learned,self.kc2kc_STM)
        #     self.dvs.set_spikes(self.pre_learning_input[0], self.pre_learning_input[1] * ms)
        #     self.net.run(self.pre_learning_input[1][-1] * ms)
        #     self.dvs.set_spikes(self.sim_input[0] , self.sim_input[1] * ms)
        #     self.net.run(self.sim_input[1][-1] * ms - self.pre_learning_input[1][-1] * ms)


