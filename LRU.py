import random
import sys
from disk_struct import Disk
from page_replacement_algorithm import  page_replacement_algorithm
from CacheLinkedList import  CacheLinkedList
import numpy as np

# sys.path.append(os.path.abspath("/home/giuseppe/))

## Keep a LRU list.
## Page hits:
##      Every time we get a page hit, mark the page and also move it to the MRU position
## Page faults:
##      Evict an unmark page with the probability proportional to its position in the LRU list.
class LRU(page_replacement_algorithm):

    def __init__(self, param):
        self.N = param#['cache_size']
        
        self.disk = CacheLinkedList(self.N)
        
        self.unique = {}
        self.unique_cnt = 0
        self.pollution_dat_x = []
        self.pollution_dat_y = []
        self.time = 0
        self.reused_block_count = 0
        self.page_entering_cache = {}
        self.unique_block_count = 0
        self.block_reused_duration = 0
        self.page_lifetime_cache = {}
        self.block_lifetime_duration = 0
        self.block_lifetime_durations = []
    
    def get_N(self) :
        return self.N
    
    def __contains__(self, q):
        return q in self.disk
    
    def visualize(self, ax):
        pass
    
    def getWeights(self):
#         return np.array([self. X, self.Y1, self.Y2,self.pollution_dat_x,self.pollution_dat_y ]).T
        return np.array([self.pollution_dat_x,self.pollution_dat_y ]).T
    def get_block_reused_duration(self):
        return self.block_reused_duration 

    def get_block_lifetime_duration(self):
        for pg in self.disk:
            self.block_lifetime_duration +=  self.time - self.page_lifetime_cache[pg]
            self.unique_block_count += 1
            self.block_lifetime_durations.append(self.time - self.page_lifetime_cache[pg])
        print("Unique no of blocks", self.unique_block_count )
        return self.block_lifetime_duration/ float(self.unique_block_count)
    
    def get_block_lifetime_durations(self):
        return self.block_lifetime_durations
    
    def getStats(self):
        d={}
        d['pollution'] = np.array([self.pollution_dat_x, self.pollution_dat_y ]).T
        return d
    
    def request(self,page) :
        self.time = self.time + 1
        page_fault = False
        if self.disk.inDisk(page) :
            self.disk.moveBack(page)
        else :
            if self.disk.size() == self.N :
                ## Remove LRU page
                lru = self.disk.getFront()
                self.block_lifetime_duration +=  self.time - self.page_lifetime_cache[lru]
                self.unique_block_count += 1
                self.block_lifetime_durations.append(self.time - self.page_lifetime_cache[lru])
                # print( "Page", cacheevict, "Lifetime", self.page_lifetime_cache[cacheevict],"At time", self.time, "Duration", self.time - self.page_lifetime_cache[cacheevict], "Block in Cache count", self.unique_block_count  )
                del self.page_lifetime_cache[lru] 
              
                self.disk.delete(lru)
            # Add page to the MRU position
            self.disk.add(page)
            self.page_lifetime_cache[page] = self.time
            page_fault = True
        
        if not page_fault and page in self.page_entering_cache :
                self.block_reused_duration +=  self.time - self.page_entering_cache[page]
                self.reused_block_count += 1
                self.page_entering_cache[page] =  self.time 

        else:
            self.page_entering_cache[page] =  self.time

        if page_fault :
            self.unique_cnt += 1
        
        self.unique[page] = self.unique_cnt
        
#         if self.time % self.N == 0:
#             pollution = 0
#             for pg in self.disk:
#                 if self.unique_cnt - self.unique[pg] >= 2*self.N:
#                     pollution += 1
#             self.pollution_dat_x.append(self.time)
#             self.pollution_dat_y.append(100*pollution / self.N)
        
        return page_fault

    def get_data(self):
        return [self.disk.get_data()]

    def get_list_labels(self) :
        return ['L']

#if __name__ == "__main__" :
#    if len(sys.argv) < 2 :
#        print("Error: Must supply cache size.")
#        print("usage: python3 [cache_size]")
#        exit(1)
#
#    n = int(sys.argv[1])
#    infile = open(sys.argv[2], 'r')
#    print("cache size ", n)
#
#    lru = LRU(n)
#    page_fault_count = 0
#    page_count = 0
#    for line in infile:
#        print("request: ", line)
#        if lru.request(line) :
#            page_fault_count += 1
#        page_count += 1
#
#
#    print("page count = ", page_count)
#    print("page faults = ", page_fault_count)
