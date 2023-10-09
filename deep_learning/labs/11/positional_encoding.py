import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def get_angles(pos, i, d_model):
  angle_rates = 1 / 10000 ** (2 * i / np.float32(d_model))
  return pos * angle_rates

def positional_encoding_wrong(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  print(angle_rads)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0:(d_model//2)] = np.sin(angle_rads[:, 0:(d_model//2)])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, (d_model//2):] = np.cos(angle_rads[:, (d_model//2):])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

def positional_encoding_old(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


n, d = 512, 512
pos_encoding = positional_encoding_wrong(n, d)
print(pos_encoding.shape)
pos_encoding = pos_encoding[0]

# Juggle the dimensions for the plot
#pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
#pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
#pos_encoding = tf.reshape(pos_encoding, (n, d))
#pos_encoding = tf.transpose(pos_encoding, (1,0))

plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.xlabel('Embedding dimensions')
plt.ylabel('Token positions')
plt.colorbar()
plt.show()

