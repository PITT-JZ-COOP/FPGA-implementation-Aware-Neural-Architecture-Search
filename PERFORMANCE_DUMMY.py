'''
Author: Weiwen Jiang, Xinyi Zhang

This class return the estimated hardware inference time.
You will need to build your own estimator
'''
import math
import networkx as nx
import sys
import random
import os
import sys
import math
from math import *
import csv
import time
import copy 
# from PARA import DESIGN_PARA
class SCHEDULE:
    def __init__(self,schedule_in, S1, G, OPT,NUM_LAYERS):
        self.schedule_in = schedule_in
        self.S1 = S1
        self.G = G
        self.OPT=OPT
        self.NUM_LAYERS=NUM_LAYERS

    # return the hardware inference time
    def schedule_run(self,layers,layersname):
        k = random.randrange(100, 1000000)
        return k
