from disk_struct import Disk
from page_replacement_algorithm import page_replacement_algorithm
from priorityqueue import priorityqueue
from CacheLinkedList import CacheLinkedList
import time
import numpy as np
import Queue
import heapq
import Queue as queue
# import matplotlib.pyplot as plt
import os

import Lirs
import LRU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# sys.path.append(os.path.abspath("/home/giuseppe/))

## Keep a LRU list.
## Page hits:
##      Every time we get a page hit, mark the page and also move it to the MRU position
## Page faults:
##      Evict an unmark page with the probability proportional to its position in the LRU list.
class LIRSLeCaR(page_replacement_algorithm):

    def __init__(self, cache_size, learning_rate=0, initial_weight=0.5, discount_rate=1, visualize=0):

        #assert 'cache_size' in param

        self.N = int(cache_size)
        if self.N < 4:
            self.N = 4
        self.H = int(self.N * 0.5)
        #self.H = int(1 * self.N * int(param['history_size_multiple'])) if 'history_size_multiple' in param else self.N
        self.learning_rate = learning_rate
        #float(param['learning_rate']) if 'learning_rate' in param else 0
        # self.learning_rate = 0.1
        self.initial_weight = initial_weight
        #float(param['initial_weight']) if 'initial_weight' in param else 0.5
        self.discount_rate = discount_rate
        #float(param['discount_rate']) if 'discount_rate' in param else 1
        self.Visualization = visualize
        #'visualize' in param and bool(param['visualize'])
        # self.discount_rate = 0.005**(1/self.N)
        np.random.seed(123)

        self.CacheRecency = CacheLinkedList(self.N)
        #param["hir_percent"] = 0.07

        self.LIRS = Lirs.Lirs(self.N)
        #algorithms.GetAlgorithm.GetAlgorithm('lirs5')(param)
        self.LRU = LRU.LRU(self.N)

        self.Hist1 = CacheLinkedList(self.H)
        self.Hist2 = CacheLinkedList(self.H)

        ## Accounting variables
        self.time = 0
        self.W = np.array([self.initial_weight, 1 - self.initial_weight], dtype=np.float32)

        self.X = []
        self.Y1 = []
        self.Y2 = []
        self.eTime = {}

        self.unique = {}
        self.unique_cnt = 0
        self.pollution_dat_x = []
        self.pollution_dat_y = []
        self.pollution_dat_y_val = 0
        self.pollution_dat_y_sum = []
        self.pollution = 0

        ## Learning Rate adaptation variables

        self.seq_len = int(1 * self.N)
        self.CacheHit = 0
        self.PreviousHR = 0.0
        self.NewHR = 0.0
        self.PreviousChangeInHR = 0.0
        self.NewChangeInHR = 0.0
        self.PreviousLR = self.learning_rate
        self.NewLR = self.learning_rate
        self.learning_rates = []
        self.reset_point = self.learning_rate
        self.SampleHR = []
        self.SAMPLE_SIZE = 500
        self.SampleHitQ = queue.Queue(maxsize=self.SAMPLE_SIZE)
        self.SampleCacheHit = 0
        self.SampleChangeInHR_SIZE = 500
        self.ChangeInHRCounter = 0
        self.SampleChangeInHR = queue.Queue(maxsize=self.SampleChangeInHR_SIZE)
        self.SampleCIR = 0

        self.info = {
            'lru_misses': 0,
            'lfu_misses': 0,
            'lru_count': 0,
            'lfu_count': 0,
        }



    ##############################################################
    ## There was a page hit to 'page'. Update the data structures
    ##############################################################
    def pageHitUpdate(self, page):
        assert page in self.CacheRecency
        assert page in self.LIRS
        assert page in self.LRU
        self.CacheRecency.moveBack(page)
        self.LIRS.request(page)
        self.LRU.request(page)

    ##########################################
    ## Add a page to cache using policy 'poly'
    ##########################################
    def addToCache(self, page):
        assert page not in self.CacheRecency
        assert page not in self.LIRS
        assert page not in self.LRU

        # self.LIRS.f3.write("Adding page %d\n" % page)

        self.CacheRecency.add(page)
        # self.LIRS.f3.write("Before\n")
        # self.LIRS.printS()
        self.LIRS.request(page)
        # self.LIRS.f3.write("After\n")
        # self.LIRS.printS()
        self.LRU.request(page)




    ######################
    ## Get LFU or LRU page
    ######################
    def selectEvictPage(self, policy):
        assert self.CacheRecency.getFront() == self.LRU.disk.getFront()
        r = self.CacheRecency.getFront()
        f = self.LIRS.HIR_list_tail.page

        assert f is not None
        assert r is not None

        pageToEvit, policyUsed = None, None
        # if r == f :
        # pageToEvit,policyUsed = r,-1
        if policy == 0:
            pageToEvit, policyUsed = r, 0
        elif policy == 1:
            pageToEvit, policyUsed = f, 1

        return pageToEvit, policyUsed

    def evictPage(self, pg):
        # self.LIRS.f3.write("removing page %d\n" % pg)
        # self.LIRS.f3.write("Before\n")
        # self.LIRS.printS()
        assert pg in self.CacheRecency
        assert pg in self.LRU
        assert pg in self.LIRS

        self.CacheRecency.delete(pg)
        self.LRU.disk.delete(pg)


        #print(self.LIRS.LIR_LRU_block.page)
        ##LIRS FUU:
        page_ref = self.LIRS.page_tbl[pg]
        page_ref.isResident = False
        if page_ref.isHIR_block:
            self.LIRS.remove_HIR_list(page_ref)
            self.LIRS.free_mem_size += 1
        else:
            page_ref.isHIR_block = True
            self.LIRS.remove_LIRS_list(page_ref)
            self.LIRS.free_mem_size += 1
            if page_ref is self.LIRS.LIR_LRU_block:
                self.LIRS.LIR_LRU_block = self.LIRS.find_last_LIR_LRU()
            self.LIRS.HIR_list_tail.isHIR_block = False
            self.LIRS.remove_HIR_list(self.LIRS.HIR_list_tail)

        self.LIRS.add_HIR_n_red(page_ref)

        # self.LIRS.f3.write("After\n")



    def addToHistory(self, poly, cacheevict):
        histevict = None
        if (poly == 0) or (poly == -1 and np.random.rand() < 0.5):
            if self.Hist1.size() == self.H:
                histevict = self.Hist1.getFront()
                assert histevict in self.Hist1
                self.Hist1.delete(histevict)
            self.Hist1.add(cacheevict)
            self.info['lru_count'] += 1
        else:
            if self.Hist2.size() == self.H:
                histevict = self.Hist2.getFront()
                assert histevict in self.Hist2
                self.Hist2.delete(histevict)
            self.Hist2.add(cacheevict)
            self.info['lfu_count'] += 1

        if histevict is not None:
            #del self.freq[histevict]
            del self.eTime[histevict]


    def request(self, page):
        page_fault = False
        self.time = self.time + 1

        # Visualization data
        if self.Visualization:
            self.X.append(self.time)
            self.Y1.append(self.W[0])
            self.Y2.append(self.W[1])

        # Adapt learning rate Here
        self.updateLearningRates()

        # Process page request
        if page in self.CacheRecency:
            page_fault = False
            self.pageHitUpdate(page) # Modify This
            self.CacheHit += 1
        else:
            # Learning step: If there is a page fault in history

            pageevict = None

            reward = np.array([0, 0], dtype=np.float32)
            if page in self.Hist1:
                pageevict = page
                self.Hist1.delete(page)
                reward[0] = -1

                self.info['lru_misses'] += 1

            elif page in self.Hist2:
                pageevict = page
                self.Hist2.delete(page)
                reward[1] = -1

                self.info['lfu_misses'] += 1

            # Update Weights
            if pageevict is not None:
                self.W = self.W * np.exp(self.learning_rate * reward)
                self.W = self.W / np.sum(self.W)

            # Remove from Cache
            if self.CacheRecency.size() == self.N:
                # Choose Policy
                act = self.chooseRandom()
                cacheevict, poly = self.selectEvictPage(act) #Modify this
                self.eTime[cacheevict] = self.time

                # Remove from Cache and Add to history
                self.evictPage(cacheevict)
                self.addToHistory(poly, cacheevict)

            self.addToCache(page)

            page_fault = True

        # Count pollution
        if page_fault:
            self.unique_cnt += 1
        self.unique[page] = self.unique_cnt

        if self.time % self.N == 0:
            self.pollution = 0
            for pg in self.CacheRecency:
                if self.unique_cnt - self.unique[pg] >= 2 * self.N:
                    self.pollution += 1

            self.pollution_dat_x.append(self.time)
            self.pollution_dat_y.append(100 * self.pollution / self.N)
        self.pollution_dat_y_val += 100 * self.pollution / self.N
        self.pollution_dat_y_sum.append(self.pollution_dat_y_val)

        self.learning_rates.append(self.learning_rate)
        return page_fault

    def getQ(self):
        lamb = 0.05
        return (1 - lamb) * self.W + lamb

    def get_N(self):
        return self.N

    def __contains__(self, q):
        return q in self.CacheRecency

    def visualize(self, ax_w, ax_h, averaging_window_size):
        lbl = []
        if self.Visualization:
            X = np.array(self.X)
            Y1 = np.array(self.Y1)
            Y2 = np.array(self.Y2)
            ax_w.set_xlim(np.min(X), np.max(X))
            ax_h.set_xlim(np.min(X), np.max(X))

            ax_w.plot(X, Y1, 'y-', label='W_lru', linewidth=2)
            ax_w.plot(X, Y2, 'b-', label='W_lfu', linewidth=1)
            # ax_h.plot(self.pollution_dat_x,self.pollution_dat_y, 'g-', label='hoarding',linewidth=3)
            # ax_h.plot(self.pollution_dat_x,self.pollution_dat_y, 'k-', linewidth=3)

            ax_h.set_ylabel('Hoarding')
            ax_w.legend(loc=" upper right")
            ax_w.set_title('LeCaR')
            pollution_sums = self.getPollutions()
            temp = np.append(np.zeros(averaging_window_size), pollution_sums[:-averaging_window_size])
            pollutionrate = (pollution_sums - temp) / averaging_window_size

            ax_h.set_xlim(0, len(pollutionrate))

            ax_h.plot(range(len(pollutionrate)), pollutionrate, 'k-', linewidth=3)

        #             lbl.append(l1)
        #             lbl.append(l2)
        #             lbl.append(l3)

        return lbl

    def getWeights(self):
        return np.array([self.X, self.Y1, self.Y2, self.pollution_dat_x, self.pollution_dat_y]).T

    #         return np.array([self.pollution_dat_x,self.pollution_dat_y ]).T

    def getPollutions(self):
        return self.pollution_dat_y_sum

    def getLearningRates(self):
        return self.learning_rates

    def getStats(self):
        d = {}
        d['weights'] = np.array([self.X, self.Y1, self.Y2]).T
        d['pollution'] = np.array([self.pollution_dat_x, self.pollution_dat_y]).T
        return d

    ############################################
    ## Choose a page based on the q distribution
    ############################################
    def chooseRandom(self):
        r = np.random.rand()
        if r < self.W[0]:
            return 0
        return 1

    def updateSamples(self):
        if self.SampleChangeInHR.full():
            self.SampleCIR -= self.SampleChangeInHR.get()
        self.SampleChangeInHR.put(self.NewChangeInHR)
        self.SampleCIR += self.NewChangeInHR

        if self.SampleHitQ.full():
            self.SampleCacheHit -= self.SampleHitQ.get()
        self.SampleHitQ.put(self.NewHR)
        self.SampleCacheHit += self.NewHR

    def updateInDeltaDirection(self, delta_LR):

        delta = 0

        if (delta_LR > 0 and self.NewChangeInHR > 0) or (delta_LR < 0 and self.NewChangeInHR < 0):
            delta = 1
        elif (delta_LR < 0 and self.NewChangeInHR > 0) or (delta_LR > 0 and self.NewChangeInHR < 0):
            delta = -1
        return delta

    def updateInRandomDirection(self):

        if self.learning_rate > 0.999:
            self.learning_rate = max(self.learning_rate - (float(self.learning_rate) / 4.0), 0.001)
            # print("After LR equal and Inside postive extreme")
        elif self.learning_rate < 0.001:
            self.learning_rate = max(self.learning_rate * 2, 0.001)
            # print("After LR equal and Inside negative extreme")

        else:
            val = np.random.rand()
            if val <= 0.5:
                self.learning_rate = min(self.learning_rate + (self.learning_rate * val), 1)

            else:
                self.learning_rate = max(self.learning_rate - (self.learning_rate * (val - 0.5)), 0.001)
            # self.learning_rate = val

    def updateLearningRates(self):

        if self.time % (self.seq_len) == 0:

            self.NewHR = round(self.CacheHit / float(self.seq_len), 6)
            self.NewChangeInHR = (self.NewHR - self.PreviousHR)

            self.updateSamples()
            delta_LR = round(self.NewLR, 5) - round(self.PreviousLR, 5)
            delta = self.updateInDeltaDirection(delta_LR)

            if delta > 0:
                self.learning_rate = min(self.learning_rate + abs(float(self.learning_rate) * delta_LR), 1)
                # print("Inside positive update",delta_LR, self.NewChangeInHR,self.learning_rate)

            elif delta < 0:
                self.learning_rate = max(self.learning_rate - abs(float(self.learning_rate) * delta_LR), 0.001)
                # print("Inside negative update", delta_LR, self.NewChangeInHR, self.learning_rate)
            elif delta == 0 and (self.NewChangeInHR <= 0):
                sample_changeHR = self.SampleCIR * self.SampleChangeInHR_SIZE
                sample_HR = self.SampleCacheHit * self.SAMPLE_SIZE

                # if  self.SampleHitQ.full() and  sample_HR <= (0.001 * self.SAMPLE_SIZE) :
                #     self.learning_rate = self.reset_point
                #     self.W = np.array([self.initial_weight,1-self.initial_weight], dtype=np.float32)

                if sample_changeHR <= (0.001 * self.SampleChangeInHR_SIZE):
                    update = np.random.choice([True, False])
                    if sample_changeHR == 0:

                        if update:
                            self.updateInRandomDirection()

                    else:
                        if update:
                            self.learning_rate = self.reset_point
                            self.W = np.array([self.initial_weight, 1 - self.initial_weight], dtype=np.float32)
                    # self.SampleCIR = 0
                    # self.SampleChangeInHR.queue.clear()

                elif self.NewChangeInHR < 0:
                    self.updateInRandomDirection()

            self.PreviousLR = self.NewLR
            self.NewLR = self.learning_rate
            self.PreviousHR = self.NewHR
            self.PreviousChangeInHR = self.NewChangeInHR
            self.CacheHit = 0

    def get_list_labels(self):
        return ['L']
