"""
BMCL BAEKSUWHAN
@author: lukious

Modified by: Aryan Vaswani
"""

# from https://github.com/Lukious/wav-to-csv/blob/master/wav2csv.py
import sys
import librosa
from scipy.io import wavfile
import pandas as pd
import numpy as np


def wav2csv(inputfile):
    input_filename = inputfile
    if input_filename[-3:] != 'wav':
        print('WARNING!! Input File format should be *.wav')
        sys.exit()

    data, samrate = librosa.load(str(input_filename))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=samrate).T, axis=0)

    X = []

    result = np.array(mfcc)

    for ele in result:
        X.append(ele)

    wavData = pd.DataFrame(X)

    # wavData.columns = ['M']

    wavData.to_csv(input_filename[:-4] + "_CSV_MONO.csv", index=False)

    print('Saved as: ' + str(input_filename[:-4]) + '_CSV_MONO.csv')
