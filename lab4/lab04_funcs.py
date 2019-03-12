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

    # reshape Z to make the partial calculation easier
    Z = np.reshape(Z, (8, 64))
    v = np.ones(8, np.float)
    Y = v.dot(Z)

    # TODO: matrix multiply to get S = M * Y
    S = np.zeros(32)
    for i in range(32):
      S[i] = np.dot(Y, M[:,i])

    # invert odd coefficients of odd subbands
    if row % 2:
      output[row] = S * modulate

  return output

def ipqmf(coefficents):
  """
  input is a buffer of coefficients computed by pqmf.
  Output array has same size as coefficients, and contains the reconstreucted audio data.
  """
  V = np.zeros(1024)
  D = load_d_taps(D_TAP_FILENAME)
  output = np.zeros(coefficents.shape)
  N = np.zeros((32,64))
  for i in range(64):
    for k in range(32):
      N[k,i] = np.cos((2*k+1)*(16+i)*np.pi/64)
  rows = coefficents.shape[0]

  for row in range(rows):
    S = coefficents[row] # input 32 new subband samples
    V = np.roll(V, 64)   # shifting
    for i in range(64):       # matrixing
      V[i] = N[:,i].dot(S)
    # build a 512 value vector U
    U = np.zeros(512)
    for i in range(8):
      for j in range(32):
        U[i*64+j] = V[i*128+j]
        U[i*64+32+j] = V[i*128+96+j]
    # window by 512 coefficients
    W = U * D
    W = np.reshape(W, (16, 32))
    v = np.ones(16, np.float)
    S = v.dot(W)
    output[row] = S

  return output