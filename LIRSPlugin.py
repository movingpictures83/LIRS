import random
import sys
from disk_struct import Disk
from page_replacement_algorithm import  page_replacement_algorithm
from CacheLinkedList import  CacheLinkedList
import numpy as np


import PyIO
import PyPluMA

import lirs2_1B
import lirs2_1
import lirs2_adaptive2
import lirs2_adaptive3
import lirs2_adaptive
import lirs2
import lirs4_2_1_1_adaptive
import lirs4_2_1
import lirs5_adaptive
import lirs5
import Lirs_adaptive
import lirsalecar5
import lirslecar3lfu_adaptive
import lirslecar3lfu
import lirslecar3lru
import lirslecar4_1lfuB
import lirslecar4_1lfuN
import lirslecar4_1lfu
import lirslecar4_2_1_1_adaptive
import lirslecar4_2_1_1
import lirslecar4_2_1
import lirslecar4_2lfu
import lirslecar4lfu
import lirslecar4lirs
import LIRSLeCAR
import Lirs



class LIRSPlugin:
  def input(self, inputfile):
        self.parameters = PyIO.readParameters(inputfile)

  def run(self):
        pass

  def output(self, outputfile):
    n = int(self.parameters["n"])
    infile = open(PyPluMA.prefix()+"/"+self.parameters["infile"], 'r')
    kind = self.parameters["kind"]
    outfile = open(outputfile, 'w')
    outfile.write("cache size "+str(n))
    if (kind == "lirs2_1B"):
        lirs = lirs2_1B.Lirs2_1(n)
    elif (kind == "lirs2_1"):
        lirs = lirs2_1.Lirs2_1(n)
    elif (kind == "lirs2_adaptive2"):
        lirs = lirs2_adaptive2.Lirs2(n)
    elif (kind == "lirs2_adaptive3"):
        lirs = lirs2_adaptive3.Lirs2(n)
    elif (kind == "lirs2_adaptive"):
        lirs = lirs2_adaptive.Lirs2(n)
    elif (kind == "lirs2"):
        lirs = lirs2.Lirs2(n)
    elif (kind == "lirs4_2_1_1_adaptive"):
        lirs = lirs4_2_1_1_adaptive.lirs4_2_1_1(n)
    elif (kind == "lirs4_2_1"):
        lirs = lirs4_2_1.lirs4_2_1(n)
    elif (kind == "lirs5_adaptive"):
        lirs = lirs5_adaptive.Lirs5(n)
    elif (kind == "lirs5"):
        lirs = lirs5.Lirs5(n)
    elif (kind == "Lirs_adaptive"):
        lirs = Lirs_adaptive.Lirs(n)
    elif (kind == "lirsalecar5"):
        lirs = lirsalecar5.lirsalecar5(n)
    elif (kind == "lirslecar3lfu_adaptive"):
        lirs = lirslecar3lfu_adaptive.lirslecar3lfu(n)
    elif (kind == "lirslecar3lfu"):
        lirs = lirslecar3lfu.lirslecar3lfu(n)
    elif (kind == "lirslecar3lru"):
        lirs = lirslecar3lru.lirslecar3lru(n)
    elif (kind == "lirslecar4_1lfuB"):
        lirs = lirslecar4_1lfuB.lirslecar4_1lfu(n)
    elif (kind == "lirslecar4_1lfuN"):
        lirs = lirslecar4_1lfuN.lirslecar4_1lfuN(n)
    elif (kind == "lirslecar4_1lfu"):
        lirs = lirslecar4_1lfu.lirslecar4_1lfu(n)
    elif (kind == "lirslecar4_2_1_1_adaptive"):
        lirs = lirslecar4_2_1_1_adaptive.lirslecar4_2_1_1(n)
    elif (kind == "lirslecar4_2_1_1"):
        lirs = lirslecar4_2_1_1.lirslecar4_2_1_1(n)
    elif (kind == "lirslecar4_2_1"):
        lirs = lirslecar4_2_1.lirslecar4_2_1(n)
    elif (kind == "lirslecar4_2lfu"):
        lirs = lirslecar4_2lfu.lirslecar4_2lfu(n)
    elif (kind == "lirslecar4lfu"):
        lirs = lirslecar4lfu.lirslecar4lfu(n)
    elif (kind == "lirslecar4lirs"):
        lirs = lirslecar4lirs.lirslecar4lirs(n)
    elif (kind == "LIRSLeCAR"):
        lirs = LIRSLeCAR.LIRSLeCaR(n)
    else:
        lirs = Lirs.Lirs(n)


    page_fault_count = 0
    page_count = 0
    for line in infile:
        line = int(line.strip())
        outfile.write("request: "+str(line))
        if lirs.request(line) :
            page_fault_count += 1
        page_count += 1

    
    outfile.write("page count = "+str(page_count))
    outfile.write("\n")
    outfile.write("page faults = "+str(page_fault_count))
    outfile.write("\n")
