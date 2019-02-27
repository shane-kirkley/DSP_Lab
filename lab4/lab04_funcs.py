#!/usr/local/bin/python3
import numpy as np
import math

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

def ipqmf(coefficents):
  raise NotImplementedError()
  """
  input is a buffer of coefficients computed by pqmf.
  Output array recons has same size as coefficients, and contains the reconstreucted audio data.
  """