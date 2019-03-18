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
  Lab assignment 1.
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
    else:
      output[row] = S

  return output

def ipqmf(coefficents):
  """
  Lab assignment 3.
  input is a buffer of coefficients computed by pqmf.
  Output array has same size as coefficients, and contains the reconstructed audio data.
  """
  V = np.zeros(1024)
  D = load_d_taps(D_TAP_FILENAME)
  output = np.zeros(coefficents.shape)
  rows = coefficents.shape[0]
  N = np.zeros((64,32))
  for i in range(64):
    for k in range(32):
      N[i,k] = np.cos((2*k+1)*(16+i)*np.pi/64)
  modulate = np.resize([1,-1], 32)

  for row in range(rows):
    S = coefficents[row] # input 32 new subband samples
    if row % 2:          # re-invert odd coefficients of odd subbands
      S = S * modulate
    V = np.roll(V, 64)   # shifting
    for i in range(64):  # matrixing
      V[i] = N[i,:].dot(S)
    U = np.zeros(512)    # build a 512 value vector U
    for i in range(8):
      for j in range(32):
        U[i*64+j] = V[i*128+j]
        U[i*64+32+j] = V[i*128+96+j]
    W = U * D            # window by 512 coefficients
    W = np.reshape(W, (16, 32))
    v = np.ones(16, np.float)
    output[row] = v.dot(W)
  return output

def ipqmf_bands(coefficents, bands):
  """
  Lab assignment 6.
  input:
    coefficients - a buffer of coefficients computed by pqmf.
    bands - bitmap array where... 
      bands[i] = 1 if band i is used in reconstruction.
      bands[i] = 0 if band i is not used in reconstruction.
  output: array of reconstructed audio data
  """
  V = np.zeros(1024)
  D = load_d_taps(D_TAP_FILENAME)
  output = np.zeros(coefficents.shape)
  rows = coefficents.shape[0]
  N = np.zeros((64,32))
  for i in range(64):
    for k in range(32):
      N[i,k] = np.cos((2*k+1)*(16+i)*np.pi/64)
  modulate = np.resize([1,-1], 32)

  for row in range(rows):
    # input 32 new subband samples as specified
    S = coefficents[row] * bands 
    if row % 2:          # re-invert odd coefficients of odd subbands
      S = S * modulate
    V = np.roll(V, 64)   # shifting
    for i in range(64):  # matrixing
      V[i] = N[i,:].dot(S)
    U = np.zeros(512)    # build a 512 value vector U
    for i in range(8):
      for j in range(32):
        U[i*64+j] = V[i*128+j]
        U[i*64+32+j] = V[i*128+96+j]
    W = U * D            # window by 512 coefficients
    W = np.reshape(W, (16, 32))
    v = np.ones(16, np.float)
    output[row] = v.dot(W)
  return output