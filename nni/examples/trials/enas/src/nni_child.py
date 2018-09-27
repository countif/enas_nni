from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import time
import fcntl
import numpy as np
import tensorflow as tf
import logging
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.cifar10.data_utils import read_data
from src.cifar10.general_child import GeneralChild
from src.cifar10.micro_child import MicroChild



class ENASBaseTrial(object):

    def __init__(self):
        return


    def get_csvaa(self, controller_total_steps, child_arc):
        valid_acc_arr = []
        for idx in range(0, controller_total_steps):
            cur_valid_acc = self.sess.run\
                (self.child_model.cur_valid_acc,
                 feed_dict={self.child_model.sample_arc: child_arc[idx]})
            valid_acc_arr.append(cur_valid_acc)
        return valid_acc_arr


    # TODO: nni.send_final_result()
    def send_reward(self, epoch, rewards):
        with open(self.child_prefix + str(epoch) + ".txt", "w") as out_file:
            fcntl.flock(out_file, fcntl.LOCK_EX)
            number = len(rewards)
            out_file.write(str(number) + "\n")
            for i in range(0, number):
                out_file.write(str(rewards[i]) + "\n")


    def receive_line(self, line):
        tokens = line.split()
        arc = []
        for j in range(0, len(tokens)):
            arc.append(int(tokens[j]))
        arc = np.array(arc)
        return arc


    # TODO nni.get_parameters()
    def receive_macro_arc(self, epoch):
        input_path = self.controller_prefix + str(epoch) + ".txt"

        while True:
            # detect new file
            if os.path.exists(input_path):
                break
            else:
                time.sleep(5)

        with open(input_path, "r") as in_file:
            fcntl.flock(in_file, fcntl.LOCK_EX)
            arr = in_file.readlines()
            number = int(arr[0].split()[0])
            child_arc = []
            for i in range(number):
                line = arr[i + 1]
                tokens = line.split()
                arc = []
                for j in range(0, len(tokens)):
                    arc.append(int(tokens[j]))
                arc = np.array(arc)
                child_arc.append(arc)
        return child_arc


    def parset_child_arch(self,child_arc):
        result_arc = []
        for i in range(0,len(child_arc)):
            arc = child_arc[i]['__ndarray__']
            result_arc.append(arc)
        return result_arc

