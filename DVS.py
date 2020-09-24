#!/usr/bin/python
# -*- coding:utf8 -*-

import os
import rospy
import rosbag
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyQt5
#import pyqtgraph as pg
from mpl_toolkits.mplot3d import Axes3D
import yaml
# import rosbag_pandas
#from PyQt5 import QtCore, QtGui, QtWidgets #works for pyqt5
# import plotly.graph_objects as go
# import plotly.express as px
import numpy as np
import pandas as pd
import scipy
from mpl_toolkits.mplot3d import Axes3D

from scipy import signal
import scipy.io

def get_events(filename):
    x = []
    y = []
    ts = []
    p = []

    bag = rosbag.Bag(filename)
    print 'input:', (bag.get_end_time() - bag.get_start_time()), 'seconds'

    for topic , msg , t in bag.read_messages(topics = ['/dvs/events'] ):
        for i in msg.events:
            x.append(i.x)
            y.append(i.y)
            ts.append(int(i.ts.secs * 1e3 + i.ts.nsecs/1e6))   ## conver time resolution from ns to ms
            #p.append(i.polarity)
    bag.close()
    return x, y, ts


def chop_frame(x_c,y_c,t_c):
    l_x = 4
    h_x = 341
    l_y = 99
    # h_y =
    list_rm_index_x = [index for (index,value) in enumerate(x_c) if (value <= l_x or value >= h_x)]
    list_rm_index_y = [index for (index,value) in enumerate(y_c) if (value <= l_y)]
    list_rm = list_rm_index_x + list_rm_index_y

    x_out = np.array(x_c)
    y_out = np.array(y_c)
    t_out = np.array(t_c)

    x_nc = np.delete(x_out, list_rm) - l_x -1
    y_nc = np.delete(y_out, list_rm) - l_y -1
    t_nc = np.delete(t_out, list_rm)

    return x_nc, y_nc, t_nc


def smooth(x,y,t):
    x = np.floor(x / 16)
    y = np.floor(y / 16)


    neuran_idx = x * 10 + y   # 336/16 = 21    160/16 = 10    (8*8)
    neuran_idx_copy = neuran_idx.copy()

    t = t - t.min()
    t_max = t.max()
    #t = np.floor(t / 1e9).astype(np.int32)

    t_copy = t.copy()
    # return neuran_idx_copy,t_copy

    neuran_num_threshold = 1
    time_threshold = 1


    neurans_acti = np.zeros((1 , 1))  # the spike train arrays
    t_ms = np.zeros((1 , 1))

    last_idx_end = 0
    length_t = len(t)

    for i in range(max(t) + 1):
        idx_begin = idx_end = last_idx_end
        while (idx_end < length_t and t[idx_end] == i):
            idx_end = idx_end + 1
        neurans_1ms , count = np.unique(neuran_idx_copy[idx_begin:idx_end] , return_counts = True)
        neurans_1ms = neurans_1ms[count >= neuran_num_threshold * time_threshold]
        t_1ms = np.ones(neurans_1ms.shape[0]) * i

        if i == 0:
            neurans_acti = neurans_1ms
            t_ms = t_1ms
        else:
            neurans_acti = np.append(neurans_acti , neurans_1ms.copy() , axis = 0)
            t_ms = np.append(t_ms , t_1ms.copy() , axis = 0)
        last_idx_end = idx_end

    return neurans_acti , t_ms


def get_input(filename):
    x, y, t = get_events(filename)
    x_c, y_c, t_c = chop_frame(x, y, t)
    idx, t = smooth(x_c, y_c, t_c)

    return idx, t

#
if __name__ == '__main__':
    input_path = '/home/le/Videos/0318normallense/v20t35/'

    files = os.listdir(input_path)
    files.sort()

    for f in files:
        if not os.path.isdir(input_path + f):
            f_name = os.path.splitext(f)[0]
        idx_sp, t_sp = get_input(input_path + f)
        scipy.io.savemat('data/angle/'+f_name+'.mat', mdict = {'idx': idx_sp, 't': t_sp})
# #
# #     x,y,t = get_events('/home/le/Videos/0219normallense/v20/NB1_2020-02-19-11-11-09.bag')
# #     #scipy.io.savemat('data/input/events.mat' , mdict = {'x':x, 'y':y, 't':t})
# #
# #     x_c,y_c,t_c = chop_frame(x,y,t)
# #     #scipy.io.savemat('data/input/chop_frame.mat' , mdict = {'x':x_c, 'y':y_c, 't':t_c})
# #
# #     idx_t = chop_frame(x_c,y_c,t_c)
# #     #scipy.io.savemat('data/input/smooth.mat' , mdict = {'x':x_s, 'y':y_s, 't':t_s})

