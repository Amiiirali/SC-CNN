'''``
	Config file for initalizing synthesize files.
'''

import os

# Link to download the dataset
url = 'https://warwick.ac.uk/fac/sci/dcs/research/tia/data/crchistolabelednucleihe/crchistophenotypes_2016_04_28.zip'

# Path that we want to put our dataset in it
dataset_path = 'Datasets'


#######################################
############### Train #################
#######################################

train_path 	   = 'CRCHistoPhenotypes_2016_04_28'
image_path	   = os.path.join(train_path, 'Tissue_Images')
center_path    = os.path.join(train_path, 'Center_Points')
stain_path	   = os.path.join(train_path, 'Stain')
detection_path = os.path.join(train_path, 'Detection')


#######################################
############### Test ##################
#######################################
