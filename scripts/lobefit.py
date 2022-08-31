# standard imports
import pandas as pd
import sys
import os
import logging
import argparse
import numpy as np
import math

# code import
from SGL.lobefitter import FitObservedLobes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prefix_chars='-')
    group1 = parser.add_argument_group('Configuration Options')
    group1.add_argument("--radioimage", dest='radioimage', type=str, 
                        help='')
    group1.add_argument("--hostimage", dest='hostimage', type=str, 
                        help='')
    options = parser.parse_args()

    if options.radioimage == None:
        parser.print_help()
        raise(Exception('Radio image required.'))

    FitObservedLobes(options.radioimage, options.hostimage)