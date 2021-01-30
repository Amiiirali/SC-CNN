fimport os
import cv2
import numpy as np
from tqdm import tqdm
from shutil import copy
from scipy.io import loadmat
import requests, zipfile, io
from multiprocessing import Pool
from dataset.ColorNormalization import steinseperation


def process(name, root, image_dir, center_dir, stain_dir):

    # Copy and rename the image
    image_path = os.path.join(root, name+'.bmp')

    copy(image_path, image_dir)
    # os.rename(os.path.join(image_dir, name+'.bmp'), image_path)

    # Process Centers
    annotation_path = os.path.join(root, name+'_detection.mat')
    annotation      = loadmat(annotation_path)

    # Change to Round to be compatible with author
    # annots['detection'] = annots['detection'].astype(int)
    center_path = os.path.join(center_dir, name + '.txt')

    with open(center_path, 'w') as f:
        for point in annotation['detection']:
            # Because it is written in Matlab
            # and Matlab start from {1} instead of {0}
            f.write(f"{(round(point[0])-1, round(point[1])-1)}\n")

    # Obtaining H-Channel
    # Vahadane color normalization --> my implementation
    # of Matlab's version
    _, _, _, stain, _ = steinseperation.stainsep(image_path, 2, 0.02)
    H_Channel         = stain[0]

    stain_path = os.path.join(stain_dir, name + '_stain.png')
    cv2.imwrite(stain_path, H_Channel)

def divide_task_for_multiprocess(num_process, names, roots, image_dir_path,
                                 center_dir_path, stain_dir_path):

    result = list()

    names = np.array(names)
    roots = np.array(roots)

    for i in range(len(names)//num_process):

        image_path  = np.repeat(image_dir_path, num_process)
        center_path = np.repeat(center_dir_path, num_process)
        stain_path  = np.repeat(stain_dir_path, num_process)

        res = np.stack((names[i*num_process:(i+1)*num_process],
                        roots[i*num_process:(i+1)*num_process],
                        image_path,
                        center_path,
                        stain_path), axis=-1)

        result.append(res)

    if len(names)%num_process != 0:

        num = len(names)%num_process

        image_path  = np.repeat(image_dir_path, num)
        center_path = np.repeat(center_dir_path, num)
        stain_path  = np.repeat(stain_dir_path, num)

        res = np.stack((names[-num:],
                        roots[-num:],
                        image_path,
                        center_path,
                        stain_path), axis=-1)

        result.append(res)

    return result


def load_dataset(num_process, dataset_dir, main_path, image_path, center_path,
                 stain_path):

    image_dir_path  = os.path.join(dataset_dir, image_path)
    center_dir_path = os.path.join(dataset_dir, center_path)
    stain_dir_path  = os.path.join(dataset_dir, stain_path)
    main_dir_path   = os.path.join(dataset_dir, main_path)

    # if not, create it
    if not os.path.isdir(image_dir_path)  or \
       not os.path.isdir(center_dir_path) or \
       not os.path.isdir(stain_dir_path):

        os.mkdir(image_dir_path)
        os.mkdir(center_dir_path)
        os.mkdir(stain_dir_path)

    names, roots = list(), list()

    for root, dirs, files in os.walk(main_dir_path, topdown=True):
        for name in sorted(files):
            if name.endswith('.bmp'):

                # Remove .bmp
                names.append(name[:-4])
                roots.append(root)

    tasks = divide_task_for_multiprocess(num_process,
                                         names,
                                         roots,
                                         image_dir_path,
                                         center_dir_path,
                                         stain_dir_path)

    with Pool(processes=num_process) as pool:

        n_work = len(tasks)
        prefix = 'Data Processing'

        for idx in tqdm(range(0, n_work), desc=prefix, dynamic_ncols=True):

            pool.starmap(process, tasks[idx])


def download_dataset(url, path):

    '''
    This function downloads the dataset and extracts the .zip file.

    Input:
          1- url:
                The link to the dataset.
          2- path:
                The path we want to extract the file in it.
    '''

    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)
