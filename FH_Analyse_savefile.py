import numpy as np
from mindaffectBCI.decoder.analyse_datasets import debug_test_dataset
from mindaffectBCI.decoder.offline.load_mindaffectBCI  import load_mindaffectBCI
import matplotlib.pyplot as plt
#%matplotlib inline
#%load_ext autoreload
#%autoreload 2
plt.rcParams['figure.figsize'] = [12, 8] # bigger default figures


# select the file to load
savefile = '..\datasets\mindaffectBCI*.txt' # use the most recent file in datasets directory

X, Y, coords = load_mindaffectBCI(savefile, stopband=None, fs_out=None)
# output is: X=eeg, Y=stimulus, coords=meta-info about dimensions of X and Y
print("EEG: X({}){} @{}Hz".format([c['name'] for c in coords],X.shape,coords[1]['fs']))                            
print("STIMULUS: Y({}){}".format([c['name'] for c in coords[:-1]]+['output'],Y.shape))
# Plot the grand average spectrum to get idea of the signal quality
from mindaffectBCI.decoder.preprocess import plot_grand_average_spectrum
plot_grand_average_spectrum(X, fs=coords[1]['fs'], ch_names=coords[-1]['coords'], log=True)

X, Y, coords = load_mindaffectBCI(savefile, stopband=(3,25,'bandpass'), fs_out=100)
# output is: X=eeg, Y=stimulus, coords=meta-info about dimensions of X and Y
print("EEG: X({}){} @{}Hz".format([c['name'] for c in coords],X.shape,coords[1]['fs']))                            
print("STIMULUS: Y({}){}".format([c['name'] for c in coords[:-1]]+['output'],Y.shape))

clsfr=debug_test_dataset(X, Y, coords, model='cca', evtlabs=('re','fe'), rank=1, tau_ms=450)

# test different inner classifier.  Here we use a Logistic Regression classifier to classify single stimulus-responses into rising-edge (re) or falling-edge (fe) responses. 
debug_test_dataset(X, Y, coords,
                   preprocess_args=dict(badChannelThresh=3, badTrialThresh=None, whiten=.01, whiten_spectrum=.1),
                   model='lr', evtlabs=('re', 'fe'), tau_ms=450, ignore_unlabelled=True)