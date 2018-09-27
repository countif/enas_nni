from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import time
import logging
import tensorflow as tf
import fcntl
from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.cifar10.general_controller import GeneralController
from src.cifar10.micro_controller import MicroController
import nni
from nni.multi_phase.multi_phase_tuner import MultiPhaseTuner

class ENASBaseTuner(MultiPhaseTuner):

    def __init__(self):
        return

    def get_csvaa(self, child_totalsteps):
        child_arc = []
        for _ in range(0, child_totalsteps):
            arc = self.sess.run(self.controller_model.sample_arc)
            child_arc.append(arc)
        return child_arc


    # TODO: nni.send_final_result()
    def send_child_macro_arc(self, epoch, child_arc):

        output_path = self.controller_prefix + str(epoch) + ".txt"
        with open(output_path, "w") as out_file:
            fcntl.flock(out_file, fcntl.LOCK_EX)
            number = len(child_arc)
            out_file.write(str(number) + "\n")
            for i in range(len(child_arc)):
                arc = child_arc[i]
                self.writearcline(arc, out_file)
        return


    def writearcline(self, arc, out_file):
        line = str(arc[0])
        arc_len = len(arc)
        for idx in range(1, arc_len, 1):
            line = line + " " + str(arc[idx])
        out_file.write(line + "\n")


    #TODO nni.get_parameters()
    def receive_reward(self, epoch):
        input_path = self.child_prefix + str(epoch) + ".txt"
        while True:
            # detect new file
            if os.path.exists(input_path):
                break
            else:
                time.sleep(5)
        with open(input_path, "r") as in_file:
            fcntl.flock(in_file, fcntl.LOCK_EX)
            arr = in_file.readlines()
            number = int(arr[0].split("\n")[0])
            valid_acc_arr = []
            for i in range(number):
                line = arr[i + 1]
                valid_acc = float(line.split("\n")[0])
                valid_acc_arr.append(valid_acc)

        return valid_acc_arr