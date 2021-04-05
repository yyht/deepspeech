
import tensorflow as tf

def build_index_table(vocab_list):
  mapping_strings = tf.constant(vocab_list)

  number_of_mapping_strings = len(vocab_list)
  the_values = tf.constant(
    [[1 if i == j else 0 for i in range(number_of_mapping_strings)] for j in range(number_of_mapping_strings)],
    dtype=tf.int32)

  # Create the table for getting indices (for the_values) from the information about the board
  tensor_table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, name="index_lookup_table")
  return tensor_table

def full_onehot_process_line_as_1d_input(string_tensor, 
                    tensor_table,
                    delimiter=" "):
  with tf.name_scope("string2id"):
    return tensor_table.lookup(
          # Split the string into an array of characters
          tf.string_split(
            [string_tensor],
            delimiter=delimiter).values)
