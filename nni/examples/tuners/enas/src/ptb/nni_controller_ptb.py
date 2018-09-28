from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle as pickle
import shutil
import sys
import time

import numpy as np
import tensorflow as tf
import logging
from src import utils
from src.utils import Logger
from src.utils import print_user_flags
from src.nni_controller import ENASBaseTuner
from src.ptb.ptb_enas_controller import PTBEnasController
from src.nni_controller import ENASBaseTuner
from src.ptb_flags import *


def build_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_name+'.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger

logger = build_logger("nni_controller_ptb")


def BuildController():
    controller_model = PTBEnasController(
        rhn_depth=FLAGS.child_rhn_depth,
        lstm_size=100,
        lstm_num_layers=1,
        lstm_keep_prob=1.0,
        tanh_constant=FLAGS.controller_tanh_constant,
        temperature=FLAGS.controller_temperature,
        lr_init=FLAGS.controller_lr,
        lr_dec_start=0,
        lr_dec_every=1000000,  # never decrease learning rate
        l2_reg=FLAGS.controller_l2_reg,
        entropy_weight=FLAGS.controller_entropy_weight,
        bl_dec=FLAGS.controller_bl_dec,
        optim_algo="adam",
        sync_replicas=FLAGS.controller_sync_replicas,
        num_aggregate=FLAGS.controller_num_aggregate,
        num_replicas=FLAGS.controller_num_replicas)
    return controller_model


def get_controller_ops(controller_model):
    """
    Args:
      images: dict with keys {"train", "valid", "test"}.
      labels: dict with keys {"train", "valid", "test"}.
    """

    controller_ops = {
        "train_step": controller_model.train_step,
        "loss": controller_model.loss,
        "train_op": controller_model.train_op,
        "lr": controller_model.lr,
        "grad_norm": controller_model.grad_norm,
        "valid_ppl": controller_model.valid_ppl,
        "optimizer": controller_model.optimizer,
        "baseline": controller_model.baseline,
        "ppl": controller_model.ppl,
        "reward": controller_model.reward,
        "entropy": controller_model.sample_entropy,
        "sample_arc": controller_model.sample_arc,
    }

    return controller_ops


class ENASTuner(ENASBaseTuner):

    def __init__(self):
        controller_total_steps = FLAGS.controller_train_steps * FLAGS.controller_num_aggregate
        logger.debug("controller_total_steps\n")
        logger.debug(controller_total_steps)
        child_steps = FLAGS.child_steps

        self.controller_prefix = CONST_CONTROLLER_PREFIX
        self.child_prefix = CONST_CHILD_PREFIX
        self.controller_model = BuildController()
        self.controller_total_steps = FLAGS.controller_train_steps * FLAGS.controller_num_aggregate

        self.graph = tf.Graph()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        self.controller_model.build_trainer()
        self.controller_ops = get_controller_ops(self.controller_model)
        hooks = []
        if FLAGS.controller_training and FLAGS.controller_sync_replicas:
            sync_replicas_hook = self.controller_ops["optimizer"].make_session_run_hook(True)
            hooks.append(sync_replicas_hook)
        self.sess = tf.train.SingularMonitoredSession(
            config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir)

        logger.debug('initlize controller_model done.')


    def controller_one_step(self,epoch, valid_loss_arr,controller_total_steps):
        logger.debug("Epoch {}: Training controller".format(epoch))
        for ct_step in xrange(controller_total_steps):
            run_ops = [
                self.controller_ops["loss"],
                self.controller_ops["entropy"],
                self.controller_ops["lr"],
                self.controller_ops["grad_norm"],
                self.controller_ops["reward"],
                self.controller_ops["baseline"],
                self.controller_ops["train_op"],
            ]
            loss, entropy, lr, gn, rw, bl, _ = self.sess.run(run_ops,feed_dict={self.controller_model.valid_loss:valid_loss_arr[ct_step]})
            controller_step = self.sess.run(self.controller_ops["train_step"])

            if ct_step % FLAGS.log_every == 0:
                log_string = ""
                log_string += "ctrl_step={:<6d}".format(controller_step)
                log_string += " loss={:<7.3f}".format(loss)
                log_string += " ent={:<5.2f}".format(entropy)
                log_string += " lr={:<6.4f}".format(lr)
                log_string += " |g|={:<10.7f}".format(gn)
                log_string += " rw={:<7.3f}".format(rw)
                log_string += " bl={:<7.3f}".format(bl)
                logger.debug(log_string)
        return



def main(_):
    logger.debug("-" * 80)

    if not os.path.isdir(FLAGS.output_dir):
        logger.debug("Path {} does not exist. Creating.".format(FLAGS.output_dir))
        os.makedirs(FLAGS.output_dir)
    elif FLAGS.reset_output_dir:
        logger.debug("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
        shutil.rmtree(FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)

    logger.debug("-" * 80)

    logger.debug('Parse parameter done.')


    controller_total_steps = FLAGS.controller_train_steps * FLAGS.controller_num_aggregate
    logger.debug("controller_total_steps\n")
    logger.debug(controller_total_steps)
    child_steps = FLAGS.child_steps

    tuner = ENASTuner()
    epoch = 0

    while True:
        if epoch >= FLAGS.num_epochs:
            break
        child_arc = tuner.get_csvaa(child_steps)
        logger.debug("child_arc length\t" + str(len(child_arc)))
        tuner.send_child_macro_arc(epoch, child_arc)
        epoch = epoch + 1
        valid_loss_arr = tuner.receive_reward(epoch)
        tuner.controller_one_step(epoch, valid_loss_arr, controller_total_steps)


if __name__ == "__main__":
  tf.app.run()