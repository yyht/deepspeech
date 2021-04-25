
import tensorflow as tf
import numpy as np
import collections, re

def init_pretrained(assignment_map, initialized_variable_names,
                    tvars, init_checkpoint, **kargs):
  if len(assignment_map) >= 1:
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    for var in tvars:
      init_string = ""
      init_checkpoint_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
        init_checkpoint_string = init_checkpoint
      
      tf.logging.info(" name = %s, shape = %s%s, from checkpoint = %s", 
              var.name, var.shape, init_string, init_checkpoint_string)
      print(" name = {}, shape = {}{}, from checkpoint = {}".format(
              var.name, var.shape, init_string, init_checkpoint_string))
  else:
    tf.logging.info(" **** no need for checkpoint initialization **** ")
    print(" **** no need for checkpoint initialization **** ")


def get_actual_scope(name, exclude_scope):
  return "/".join([exclude_scope, name])

def get_assigment_map_from_checkpoint(tvars, init_checkpoint, **kargs):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  exclude_scope = kargs.get("exclude_scope", "")

  restore_var_name = kargs.get("restore_var_name", [])
  for name in restore_var_name:
    tf.logging.info("== restore variable name from checkpoint: %s ==", name)
    print("== restore variable name from checkpoint: {} ==".format(name))

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  init_vars_name_list = []
  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    init_vars_name_list.append(name)
    if len(exclude_scope) >= 1:
      assignment_name = get_actual_scope(name, exclude_scope)
    else:
      assignment_name = name

    if assignment_name not in name_to_variable:
      continue
    else:
      if np.prod(var) != np.prod(name_to_variable[assignment_name].shape):
        continue

    if len(restore_var_name) >= 1:
      if name not in restore_var_name:
        continue

    assignment_map[name] = assignment_name
    initialized_variable_names[assignment_name] = 1
    initialized_variable_names[assignment_name + ":0"] = 1

  for name in name_to_variable:
    if name not in initialized_variable_names and name in init_vars_name_list:
      if len(restore_var_name):
        if name in restore_var_name:
          assignment_map[name] = name
          initialized_variable_names[name] = 1
          initialized_variable_names[name + ":0"] = 1
      else:
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)

def load_multi_pretrained(var_checkpoint_dict_list, **kargs):
  print(kargs.get("exclude_scope", ""), "===============")
  def init_multi_model(var_checkpoint_dict_list):
    for item in var_checkpoint_dict_list:
      tvars = item['tvars']
      init_checkpoint = item['init_checkpoint']
      exclude_scope = item['exclude_scope']
      restore_var_name = item.get('restore_var_name', [])
      [assignment_map, 
      initialized_variable_names] = get_assigment_map_from_checkpoint(
                                tvars, 
                                init_checkpoint, 
                                exclude_scope=exclude_scope,
                                restore_var_name=restore_var_name)
      init_pretrained(assignment_map, 
                    initialized_variable_names,
                    tvars, init_checkpoint, **kargs)

  scaffold_fn = None
  if kargs.get('use_tpu', 0) == 0:
    init_multi_model(var_checkpoint_dict_list)
  else:
    tf.logging.info(" initializing parameter from init checkpoint ")
    print(" initializing parameter from init checkpoint ")
    def tpu_scaffold():
      init_multi_model(var_checkpoint_dict_list)
      return tf.train.Scaffold()
    scaffold_fn = tpu_scaffold
  return scaffold_fn