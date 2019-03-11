import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import lab04_funcs as lab4

# generate plots of coefficients for 5 seconds of all audio data

fs, data = wavfile.read(sys.argv[1])
data = data[:fs*5]
coeffs = lab4.pqmf(data)

plt.plot(coeffs.T.flatten())
plt.show()