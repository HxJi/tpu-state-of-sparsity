"""Calculates weight sparsity for a model checkpoint."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
import re

flags.DEFINE_string(
    "checkpoint",
    None,
    "Path to checkpoint."
)

def get_sparsity(checkpoint, suffixes):
  """Helper function to calculate and print sparsity from a checkpoint.

  Args:
    checkpoint: path to checkpoint.
    suffixes: possible suffixes of mask variables in the checkpoint.
    mask_fn: helper function to calculate the weight mask from a saved
      tensor.
  """
  ckpt_reader = tf.train.NewCheckpointReader(checkpoint)

  # Create a list of variable names to process.
  all_names = ckpt_reader.get_variable_to_shape_map().keys()
  # Gather all variables ending with the specified suffixes
  tensor_names = []
  tensor_order = []
  for s in suffixes:
    tensor_names += [x for x in all_names if x.endswith(s) and x.startswith('conv')]

  tensor_names.sort(key=natural_keys)
  nnz = 0.0
  total = 0.0
  for s in tensor_names:
    tensor = ckpt_reader.get_tensor(s)
    mask = mask_fn(tensor)
    print(1-(np.count_nonzero(mask)/mask.size))
    nnz += np.count_nonzero(mask)
    total += mask.size


def main(_):
  flags.mark_flag_as_required("checkpoint")

if __name__ == "__main__":
  app.run(main)
