from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from gd1 import *


# figures




# calculations

def gd1_width(sig=12*u.arcmin, d=8*u.kpc):
    """Calculate stream width in physical units"""
    
    print((np.arctan(sig.to(u.radian).value)*d).to(u.pc))

def gd1_length():
    """Calculate stream length in physical units"""

def width_range():
    """Find width at different locations along the stream"""
    
    #pickles here: /home/ana/projects/GD1-DR2/notebooks/stream-probs/
