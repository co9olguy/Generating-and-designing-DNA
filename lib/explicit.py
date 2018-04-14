import tensorflow as tf
import numpy as np

def match(sequence, pattern, stride=1, padding="VALID", data_format="NCHW"):
  """
  Computes the convolution of `pattern` with `sequence`.
  If `sequence` and `pattern` have the same length, just a single inner product is computed.

  :param sequence: should be of shape (batch_size, sequence_len, vocab_size) ["NHWC"]
  :param pattern: should be of shape (pattern_len, vocab_size)
  :returns: a tensor of shape (batch_size, num_activations, 1)
  """
  filter = tf.cast(tf.expand_dims(pattern, 2), tf.float32)
  if data_format=="NCHW":
    seq = tf.transpose(sequence, [0,2,1])
  elif data_format=="NHWC":
    seq = sequence
  else:
    raise Exception("Unrecognized data_format")
  seq = tf.cast(seq, tf.float32)
  conv = tf.nn.conv1d(seq, filter, stride=stride, padding=padding, data_format=data_format)

  if data_format=="NCHW":
    return tf.transpose(conv, [0,2,1])
  else:
    return conv

def max_match(sequence, pattern, stride=1, padding="VALID", data_format="NCHW"):
  """
  Computes the convolution of `pattern` with `sequence`, followed by max-pooling the activations over the sequence length.

  :param sequence: should be of shape (batch_size, sequence_len, vocab_size) ["NHWC"]
  :param pattern: should be of shape (pattern_len, vocab_size)
  """
  m = match(sequence, pattern, stride, padding, data_format)
  m = tf.expand_dims(m,1)
  with tf.Session() as sess:
    seq_len = sess.run(tf.shape(m)[2])

  pool = tf.nn.max_pool(m, ksize=[1,1,seq_len,1], strides=[1,1,1,1], padding='VALID')

  return tf.squeeze(pool, [1,2,3])



if __name__=="__main__":
  sess = tf.Session()

  I = np.eye(4)
  s = np.vstack(9*[np.expand_dims(np.vstack([I[0] for idx in np.random.choice(4, 5)]),0)])
  p = np.array(5*[[1.,0.,0.,0.]])

  m = sess.run(max_match(s,p, data_format="NHWC"))
  print(m, "<--should be all 5s")
  print("Checks completed")