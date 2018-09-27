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

from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.utils import print_user_flags

from src.ptb.ptb_enas_child import PTBEnasChild
from src.ptb.ptb_enas_controller import PTBEnasController

flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")
DEFINE_string("search_for", None, "[rhn|base|enas]")

DEFINE_string("child_fixed_arc", None, "")
DEFINE_integer("batch_size", 25, "")
DEFINE_integer("child_base_number", 4, "")
DEFINE_integer("child_num_layers", 2, "")
DEFINE_integer("child_bptt_steps", 20, "")
DEFINE_integer("child_lstm_hidden_size", 200, "")
DEFINE_float("child_lstm_e_keep", 1.0, "")
DEFINE_float("child_lstm_x_keep", 1.0, "")
DEFINE_float("child_lstm_h_keep", 1.0, "")
DEFINE_float("child_lstm_o_keep", 1.0, "")
DEFINE_boolean("child_lstm_l_skip", False, "")
DEFINE_float("child_lr", 1.0, "")
DEFINE_float("child_lr_dec_rate", 0.5, "")
DEFINE_float("child_grad_bound", 5.0, "")
DEFINE_float("child_temperature", None, "")
DEFINE_float("child_l2_reg", None, "")
DEFINE_float("child_lr_dec_min", None, "")
DEFINE_float("child_optim_moving_average", None,
             "Use the moving average of Variables")
DEFINE_float("child_rnn_l2_reg", None, "")
DEFINE_float("child_rnn_slowness_reg", None, "")
DEFINE_float("child_lr_warmup_val", None, "")
DEFINE_float("child_reset_train_states", None, "")
DEFINE_integer("child_lr_dec_start", 4, "")
DEFINE_integer("child_lr_dec_every", 1, "")
DEFINE_integer("child_avg_pool_size", 1, "")
DEFINE_integer("child_block_size", 1, "")
DEFINE_integer("child_rhn_depth", 4, "")
DEFINE_integer("child_lr_warmup_steps", None, "")
DEFINE_string("child_optim_algo", "sgd", "")

DEFINE_boolean("child_sync_replicas", False, "")
DEFINE_integer("child_num_aggregate", 1, "")
DEFINE_integer("child_num_replicas", 1, "")

DEFINE_float("controller_lr", 1e-3, "")
DEFINE_float("controller_lr_dec_rate", 1.0, "")
DEFINE_float("controller_keep_prob", 0.5, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_tanh_constant", None, "")
DEFINE_float("controller_temperature", None, "")
DEFINE_float("controller_entropy_weight", None, "")
DEFINE_float("controller_skip_target", None, "")
DEFINE_float("controller_skip_rate", None, "")

DEFINE_integer("controller_num_aggregate", 1, "")
DEFINE_integer("controller_num_replicas", 1, "")
DEFINE_integer("controller_train_steps", 50, "")
DEFINE_integer("controller_train_every", 2,
               "train the controller after how many this number of epochs")
DEFINE_boolean("controller_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("controller_training", True, "")

DEFINE_integer("num_epochs", 300, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")


def get_ops(child_model,controller_model):
  """Create relevant models."""
  ops = {}

  if FLAGS.search_for == "enas":
    assert FLAGS.child_lstm_hidden_size % FLAGS.child_block_size == 0, (
      "--child_block_size has to divide child_lstm_hidden_size")
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
  else:
    raise ValueError("Unknown search_for {}".format(FLAGS.search_for))
  child_ops = {
    "global_step": child_model.global_step,
    "loss": child_model.loss,
    "train_op": child_model.train_op,
    "train_ppl": child_model.train_ppl,
    "train_reset": child_model.train_reset,
    "valid_reset": child_model.valid_reset,
    "test_reset": child_model.test_reset,
    "lr": child_model.lr,
    "grad_norm": child_model.grad_norm,
    "optimizer": child_model.optimizer,
    "num_train_batches": child_model.num_train_batches,
    "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
    "eval_func": child_model.eval_once,
  }


  return controller_ops,child_ops






def getcontrollerlog(sess,controller_ops):
  print("Here are 10 architectures")
  for _ in xrange(10):
    arc, rw = sess.run([
      controller_ops["sample_arc"],
      controller_ops["reward"],
    ])
    print("{} rw={:<.3f}".format(np.reshape(arc, [-1]), rw))
  return
def getarchlog(child_ops,sess,epoch,best_valid_ppl):
  print("Epoch {}: Eval".format(epoch))
  valid_ppl = child_ops["eval_func"](sess, "valid")
  if valid_ppl < best_valid_ppl:
    best_valid_ppl = valid_ppl
    sess.run(child_ops["test_reset"])
    child_ops["eval_func"](sess, "test", verbose=True)

  sess.run([
    child_ops["train_reset"],
    child_ops["valid_reset"],
    child_ops["test_reset"],
  ])
  total_tr_ppl = 0
  num_batches = 0

  print("-" * 80)
  return num_batches,total_tr_ppl,best_valid_ppl


def train(mode="train"):
  assert mode in ["train", "eval"], "Unknown mode '{0}'".format(mode)

  with open(FLAGS.data_path) as finp:
    x_train, x_valid, x_test, _, _ = pickle.load(finp)
    print("-" * 80)
    print("train_size: {0}".format(np.size(x_train)))
    print("valid_size: {0}".format(np.size(x_valid)))
    print(" test_size: {0}".format(np.size(x_test)))

  g = tf.Graph()
  with g.as_default():
    child_model = BuildChild(x_train, x_valid, x_test)
    controller_model = BuildController()
    child_model.connect_controller()
    controller_model.build_trainer()

    child_ops,controller_ops = get_ops(child_model, controller_model)

    print("-" * 80)
    print("Starting session")
    with tf.train.SingularMonitoredSession(
      hooks=None, checkpoint_dir=FLAGS.output_dir) as sess:
        start_time = time.time()

        if mode == "eval":
          sess.run(child_ops["valid_reset"])
          child_ops["eval_func"](sess, "valid", verbose=True)
          sess.run(child_ops["test_reset"])
          child_ops["eval_func"](sess, "test", verbose=True)
          sys.exit(0)

        num_batches = 0
        total_tr_ppl = 0
        best_valid_ppl = 67.00
        total_steps =  FLAGS.controller_train_every * child_ops["num_train_batches"]
        controller_total_steps = FLAGS.controller_train_steps * FLAGS.controller_num_aggregate

        while True:


          valid_acc_arr = []
          for idx in range(0,controller_total_steps):
            cur_valid_acc = sess.run(child_model.cur_valid_acc)
            valid_acc_arr.append(cur_valid_acc)

          if actual_step % total_steps == 0:
            if (FLAGS.controller_training and epoch % FLAGS.controller_train_every == 0):
              ChildReset(sess, child_ops)
              ControllerOneStep(controller_model=controller_model,epoch=epoch,controller_ops=controller_ops,sess=sess,valid_acc_arr=valid_acc_arr)
              getcontrollerlog(sess, controller_ops)

          ChildReset(sess, child_ops)
          num_batches, total_tr_ppl, best_valid_ppl = getarchlog(child_ops, sess, epoch,best_valid_ppl)
          num_batches = 0
          total_tr_ppl = 0
          if epoch >= FLAGS.num_epochs:
            child_ops["eval_func"](sess, "test", verbose=True)
            break

def main(_):
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print("-" * 80)
  log_file = os.path.join(FLAGS.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)

  utils.print_user_flags()
  train(mode="train")


if __name__ == "__main__":
  tf.app.run()

