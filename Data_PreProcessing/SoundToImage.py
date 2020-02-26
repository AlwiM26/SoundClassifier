import numpy as np
import librosa
from librosa import display
from matplotlib import pyplot as plt
import os

# Directory for the .WAV file
rootdir = '/dir' # Put your dataset dir here

# counter for the file name
counter = 0

# First loop through all the class name (each class is stored in different directory)
for dir in os.listdir(rootdir):
    # In the first directory loop through all the WAV file
    # The directory variable is used to store the all wav file of each class
    directory = rootdir + '/' + dir
    
    # Than loop through all the file in the current directory
    for filename in os.listdir(directory):       
        if filename.endswith(".wav"):            
            y, sr = librosa.load(os.path.join(directory, filename), sr=None) # Load the wav sound
            ps = librosa.feature.melspectrogram(y=y, sr=sr)
            librosa.display.specshow(librosa.power_to_db(ps, ref=np.max))

            #  adjust the image to remove the white padding in the spectrogram image
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
            # Save the spectrogram image to the datasets file into the class (directory name)                    
            plt.savefig('/output/{}/{}.png'.format(dir, counter)) # Put the output folder path at the output
        else:
            continue
        counter += 1
    counter = 0
