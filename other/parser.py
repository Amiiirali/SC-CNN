import argparse
import os

def parse_input():

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, required=True,
                        help="""<Required> Mode should be either train or test
                        !""")

    parser.add_argument('--version', type=int, required=True,
                        help="""<Required> Mode should be either 1 (original
                        paper) or !""")

    parser.add_argument('--patch_size', nargs='+', type=int, required=True,
                        help="""<Required> Patch Size for cropping original
                        image. In the paper it is suggested to be [27*27]. It
                        should be in 2 numbers: [H, W], which should inserted
                        as: H W !""")

    parser.add_argument('--stride_size', nargs='+', type=int, required=True,
                        help="""<Required> Stride Size, it should be in 2
                        numbers: [stride_H, stride_W], which should inserted as:
                        stride_H stride_W !""")

    parser.add_argument('--heatmap_size', nargs='+', type=int, required=True,
                        help="""<Required> HeatMap Size, it should be in 2
                        numbers: [out_H, out_W], which should inserted as:
                        out_H out_W !""")

    parser.add_argument('--d', type=int, default=4, required=False, help="""
                        Distance from each nucleus center that effect the
                        heatmap!""")

    parser.add_argument('--valid_coeff', type=int, default=0.2, required=False,
                        help="""Validation Coefficient which
                        determines how amount of training data should be saved
                        for validation!""")

    parser.add_argument('--batch_size', type=int, default=64, required=False,
                        help="""The Batch Size!""")

    parser.add_argument('--num_workers', type=int, default=4, required=False,
                        help="""The number of workers!""")

    parser.add_argument('--M', type=int, default=1, required=False,
                        help="""Maximum Number of Center in each patch!""")

    parser.add_argument('--momentum', type=int, default=0.9, required=False,
                        help="""Momentum!""")

    parser.add_argument('--lr', type=float, default=0.001, required=False,
                        help="""Learnin Rate!""")

    parser.add_argument('--epochs', type=int, default=100, required=False,
                        help="""Epoch Number!""")

    parser.add_argument('--gpu', type=int, default=[0], nargs='+', help="""Used
                        gpu""")

    parser.add_argument('--save', type=str, default='CellPoint.pt', help=""
                        "Name of the saved weights""")

    parser.add_argument('--load', type=str, help=""
                        "Name of the loaded weights""")

    parser.add_argument('--log_dir', type=str, help=""
                        "Name of the directory of logs""")

    parser.add_argument('--auto_grad', type=str, default='PyTorch', help=""
                        "the gradient calculated by Amirali's implementation or PyTorch""")

    parser.add_argument('--verbose', action='store_true', default=False, help=""
                        "Print batch loss""")

    parser.add_argument('--test_dir', type=str, help=""
                        "path to test folder images""")

    args = parser.parse_args()

    return args
