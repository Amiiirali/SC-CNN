from matplotlib import pyplot as plt
import argparse
from other import utils
from PIL import Image
from numpy import asarray


def parse_input():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                        help="""<Required> path to image!""")
    parser.add_argument('--center', type=str, required=True,
                        help="""<Required> path to center!""")
    parser.add_argument('--label', type=str, required=False,
                        help="""<Required> path to true center!""")
    args = parser.parse_args()
    return args


def show_cells(img_path, center_path, label_path):

    img = asarray(Image.open(img_path))
    center_txt = open(center_path, "r")
    cells = utils.read_center_txt(center_txt)

    if label_path:
        label_txt = open(label_path, "r")
        truth_cells = utils.read_center_txt(label_txt)

    plt.imshow(img, interpolation='nearest')
    plt.scatter(cells[:,0], cells[:,1], c='r', s=2)
    if label_path:
        plt.scatter(truth_cells[:,0], truth_cells[:,1], c='b', s=2)
    plt.show()

if __name__ == "__main__":

    arg = parse_input()
    show_cells(arg.image, arg.center, arg.label)
