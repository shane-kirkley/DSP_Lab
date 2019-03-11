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
  raise NotImplementedError()
  """
  Input is a buffer (numpy array) that contains an integer number of frames of audio data.
  Output coefficients has the same size as input buffer and contains the subband coefficents.
  Each frame will contain 576 = 18×32 audio samples. The output of a frame will be
  a sequence of 18 vectors of sub-band coefficients, where each vector has size 32
  """
  # input is an array of some number of frames
  # Should input be pre-shaped into frames?
  frameSize = 576
  nFrames = np.floor(len(input) / frameSize)

  # output should be organized as:
  # S_0[0] ... S_0[Ns-1] ... S_31[0] ...S_31[Ns-1]
  # where S_i[k] is the coeff from subband i computed for packet k of 32 audio samples.
  output = np.zeros(nFrames * 18 * 32)
  
  # filtering is performed on a buffer X of size 512
  X = np.zeros(512)

  C = load_c_taps(C_TAP_FILENAME)
  
  for frame in range(nFrames):
    offset = frame * frameSize
    # loop over 18 non overlapping blocks of size 32
    for index in range(18):
      # process a block of 32 new input samples
      # see flow chat in fig. 2
      
      # shift X right by 32
      np.roll(X, 32)
      # bring in 32 audio samples
      block_index = offset + index*32
      X[0 : 32] = input[block_index : block_index + 32]
      # window by 512 coeff, produce vector Z
      Z = C * X
      # partial calculation - 64 elements of Y (???)

      # undo the frequency inversion.




  # for i = 511 down to 32, do X[i] = X[i-32]
  # Window by 512 Coefficients to produce vector Z:
  # for i = 0 to 511, do Z[i] = C[i] * X[i]
  # partial calculation: for i =0 to 63, do Y[i] = sum(j=0to7) of Z[i] + 64j
  # Calculate 32 samples by matrixing:
  # for i = 0 to 63 so S[i] = sum(k=0 to 63) of M[i,k]*Y[k]
  # Output 32 subband samples
  
  

def ipqmf(coefficents):
  raise NotImplementedError()
  """
  input is a buffer of coefficients computed by pqmf.
  Output array recons has same size as coefficients, and contains the reconstreucted audio data.
  """