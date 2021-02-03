'''
This is the implementation of following paper:

    @inproceedings{
    Author = {Korsuk Sirinukunwattana and Shan E Ahmed Raza and Yee-Wah Tsang and David R. J. Snead and Ian A. Cree and Nasir M. Rajpoot},
    Booktitle = {IEEE TRANSACTIONS ON MEDICAL IMAGING},
    Title = {Locality Sensitive Deep Learning for Detection and Classification of Nuclei in Routine Colon Cancer Histology Images},
    Year = {2016}}


Amirali Darbandsari
AIM Lab
'''


import os
from other.parser import parse_input
from dataset.dataset import dataset
from preprocess.preprocess import MakeH5File
from train import Train
from test import Test

if __name__ == "__main__":

    arg = parse_input()

    # Download dataset
    dataset(arg)

    if arg.mode == 'train':

        # PreProcess
        root = os.path.dirname(os.path.realpath(__file__))
        h5 = MakeH5File(root, arg)
        h5.run()
        # Train
        train = Train(arg)
        train.run()

    if arg.mode == 'test':
        test = Test(arg)
        test.run()
