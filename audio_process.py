import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_spect(x):
    stft = np.abs(librosa.stft(x, n_fft=1024, hop_length=160, win_length=400, window='hamming'))
    spect = librosa.amplitude_to_db(stft)
    return spect

def show_spect(spect, fs, file):
    librosa.display.specshow(spect, sr=fs)
    #plt.colorbar()
    plt.savefig(file)

dir_list = os.listdir('data')

for d in dir_list:
    x, fs = librosa.load(os.path.join('data', d, d+'.wav'), sr=16000, duration=3)
    spect = calculate_spect(x)
    shape = spect.shape
    print(shape)

    normlized_spect = np.array([])


    #周波数ごとに正規化（たぶん）
    for sp in spect:
        mean = np.mean(sp)
        std = np.std(sp)
        norm_sp = (sp - mean) / std

        normlized_spect = np.append(normlized_spect, norm_sp)

    normlized_spect = np.reshape(normlized_spect, shape)
    np.save(os.path.join('data', d, d), normlized_spect)