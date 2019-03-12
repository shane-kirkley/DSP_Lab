import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import lab04_funcs as lab4

fs, data = wavfile.read(sys.argv[1])
data = data[:fs*5] # trim to 5 seconds

coeffs = lab4.pqmf(data)

# plt.plot(coeffs.T.flatten(), linewidth=0.5)
# plt.show()

synth = lab4.ipqmf(coeffs)
synth = synth.flatten()

#plt.subplot(2,1,1)
plt.plot(data, linewidth=0.5)
#plt.subplot(2,1,2)
plt.plot(synth.flatten(), linewidth=0.5)
plt.show()

synth[:-512] = synth[512:]
synth = synth[:512]
data = data[:synth.shape[0]]

diff = data - synth
err = np.max(diff)
print(err)