#!/usr/local/bin/python3
import numpy as np
import math

C_TAP_FILENAME = "the_c_taps.txt"
D_TAP_FILENAME = "the_d_taps.txt"

def load_c_taps( filename_path ):
  fd = open(filename_path, 'r')
  c = np.zeros(512, np.float)
  for ctr, value in enumerate(fd):
    c[ctr] = np.float( value )
  fd.close()
  return c

def load_d_taps( filename_path ):
  fd = open(filename_path, 'r')
  d = np.zeros(512, np.float)
  for ctr, value in enumerate(fd):
    d[ctr] = np.float( value )
  fd.close()
  return d

def pqmf(input):
  """
  Input is a buffer of audio data with integer multiple of 32 (trimmed if not).
  Output coefficients has the same size as input buffer and contains the subband coefficents.
  """
  X = np.zeros(512)
  C = load_c_taps(C_TAP_FILENAME)
  M = np.zeros((64,32))
  for k in range(32):
    for r in range(64):
      M[r,k] = np.cos((2*k+1)*(r-16)*np.pi/64)

  # sort input into 32 columns
  rows = int(np.floor(len(input)/32))
  input = input[:rows*32] # trim extra samples
  input = np.reshape(input, (rows,32))

  # output same size as input
  output = np.zeros(input.shape)
  
  modulate = np.resize([1,-1], 32)

  for row in range(rows):
    X = np.roll(X, 32)
    X[:32] = np.flip(input[row])
    Z = C * X
    Y = 8 * Z[:64] + 1792
    S = np.zeros(32)
    for i in range(32):
      S[i] = np.dot(Y, M[:,i])
    output[row] = S * modulate

  return output

def ipqmf(coefficents):
  raise NotImplementedError()
  """
  input is a buffer of coefficients computed by pqmf.
  Output array recons has same size as coefficients, and contains the reconstreucted audio data.
  """