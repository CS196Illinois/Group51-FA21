"""
BMCL BAEKSUWHAN
@author: lukious

Modified by: Aryan Vaswani
"""

# from https://github.com/Lukious/wav-to-csv/blob/master/wav2csv.py
import sys, os, os.path
from scipy.io import wavfile
import pandas as pd


def wav2csv(inputfile):
    input_filename = inputfile
    if input_filename[-3:] != 'wav':
        print('WARNING!! Input File format should be *.wav')
        sys.exit()

    samrate, data = wavfile.read(str(input_filename))

    wavData = pd.DataFrame(data)

    wavData.columns = ['M']

    wavData.to_csv(str(input_filename[:-4] + "_CSV_MONO.csv"), mode='w')

    print('Saved as: ' + str(input_filename[:-4]) + '_CSV_MONO.csv')
