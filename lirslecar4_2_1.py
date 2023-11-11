from disk_struct import Disk
from page_replacement_algorithm import page_replacement_algorithm
from priorityqueue import priorityqueue
from CacheLinkedList import CacheLinkedList
import time
import numpy as np
import Queue
import heapq
import Queue as queue
from collections import OrderedDict
# import matplotlib.pyplot as plt
import os
import lirs4_2_1
import LFU

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# sys.path.append(os.path.abspath("/home/giuseppe/))

## Keep a LRU list.
## Page hits:
##      Every time we get a page hit, mark the page and also move it to the MRU position
## Page faults:
##      Evict an unmark page with the probability proportional to its position in the LRU list.
class lirslecar4_2_1(page_replacement_algorithm):
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

        #self.CacheRecency = CacheLinkedList(self.N)
        #param["hir_percent"] = 0.07

        self.LIRS = lirs4_2_1.lirs4_2_1(self.N, 0.07)
        #algorithms.GetAlgorithm.GetAlgorithm('lirs5')(param)
        self.LFUalg = LFU.LFU(self.N)


        self.Hist1 = OrderedDict()

        self.log = False

        ## Accounting variables
        self.time = 0
        self.timer = 0
        self.W = np.array([self.initial_weight, 1 - self.initial_weight], dtype=np.float32)

        self.X = []
        self.Y1 = []
        self.Y2 = []
        self.eTime = {}

        self.unique = {}
        self.unique_cnt = 0
        self.reused_block_count = 0
        self.page_entering_cache = {}
        self.unique_block_count = 0
        self.block_reused_duration = 0
        self.page_lifetime_cache = {}
        self.block_lifetime_duration = 0
        self.block_lifetime_durations = []
        self.pollution_dat_x = []
        self.pollution_dat_y = []
        self.pollution_dat_y_val = 0
        self.pollution_dat_y_sum = []
        self.pollution = 0

        ## Learning Rate adaptation variables
        self.reset_point = self.learning_rate
        # self.reset_point = 0.45
        self.seq_len = int(1 * self.N)
        ### PLot with self.seq_len = int( 2 * self.N )
        self.reset_seq_len = int(self.N)
        self.CacheHit = 0
        self.PreviousHR = 0.0
        self.NewHR = 0.0
        self.PreviousChangeInHR = 0.0
        self.NewChangeInHR = 0.0
        self.learning_rate = np.sqrt(2 * np.log(2) / self.N)
        self.reset_point = min(1, max(0.001, self.learning_rate))
        self.max_val = min(1, max(0.001, self.learning_rate))
        self.PreviousLR = 0.0
        # self.PreviousLR = 0.0
        self.NewLR = self.learning_rate
        self.learning_rates = []

        self.SampleHR = []

        self.SampleCIR = 0
        self.hitrate_negative_counter = 0
        self.hitrate_zero_counter = 0

        self.LRU = 0
        self.LFU = 1
        self.page_fault_count = 0

        self.info = {
            'lru_misses': 0,
            'lfu_misses': 0,
            'lru_count': 0,
            'lfu_count': 0,
        }

    def get_N(self):
        return self.N

    def __contains__(self, q):
        return q in self.LFUalg

    def getWeights(self):
        return np.array([self.X, self.Y1, self.Y2, self.pollution_dat_x, self.pollution_dat_y]).T

    #         return np.array([self.pollution_dat_x,self.pollution_dat_y ]).T

    def getPollutions(self):
        return self.pollution_dat_y_sum

    def getLearningRates(self):
        return self.learning_rates

    def get_block_reused_duration(self):
        return self.block_reused_duration

    def get_block_lifetime_duration(self):
        for pg in self.LFUalg:
            self.block_lifetime_duration += self.time - self.page_lifetime_cache[pg]
            self.unique_block_count += 1
            self.block_lifetime_durations.append(self.time - self.page_lifetime_cache[pg])
        print("Unique no of blocks", self.unique_block_count)
        return self.block_lifetime_duration / float(self.unique_block_count)

    def get_block_lifetime_durations(self):
        return self.block_lifetime_durations

    def getStats(self):
        d = {}
        d['weights'] = np.array([self.X, self.Y1, self.Y2]).T
        d['pollution'] = np.array([self.pollution_dat_x, self.pollution_dat_y]).T
        return d

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

        return lbl

    ##############################################################
    ## There was a page hit to 'page'. Update the data structures
    ##############################################################
    def pageHitUpdate(self, page):
        assert page in self.LIRS
        assert page in self.LFUalg

        self.LIRS.request(page, False, 0)
        self.LFUalg.request(page)

        if self.log:
            self.LIRS.f3.write("Page HIT\n")
            self.LIRS.f3.write("_________________________________________________________________________________________\n")


    ##########################################
    ## Add a page to cache using policy 'poly'
    ##########################################
    def addToCache(self, page, saved_freq, from_lfu_hist):
        assert page not in self.LIRS
        assert page not in self.LFUalg

        # if(poly == 1):
        #     self.LIRS.request(page)
        if from_lfu_hist:
            self.LIRS.request(page, True, saved_freq+1)
        else:
            self.LIRS.request(page, False, 0)

        self.LFUalg.request(page)
        self.LFUalg.PQ.increase(page, saved_freq)

        if self.log:
            self.LIRS.f3.write("END\n")
            self.LIRS.f3.write("_________________________________________________________________________________________\n")




    ######################
    ## Get LFU or LFU page
    ## 0 lfu
    ## 1 lirs
    ######################
    def selectEvictPage(self, policy):
        r = self.LFUalg.PQ.peaktop()
        f = next(iter(self.LIRS.residentHIRList))

        assert f is not None
        assert r is not None

        pageToEvit, policyUsed = None, None
        if r == f :
            pageToEvit,policyUsed = r,-1
        if policy == 0:
            pageToEvit, policyUsed = r, 0
        elif policy == 1:
            pageToEvit, policyUsed = f, 1

        if self.log:
            self.LIRS.f3.write("Page and Policy\n")
            if policyUsed == 0:
                p = "LFU"
            elif policyUsed == 1:
                p = "LIRS"
            else:
                p = "BOTH"
            self.LIRS.f3.write("page to evict: <%d>   Policy used: <%s>\n" % (pageToEvit, p))
        return pageToEvit, policyUsed

    def evictPage(self, pg, poly):
        if self.log:
            self.LIRS.f3.write("removing page %d\n" % pg)
            # self.LIRS.f3.write("Before\n")
            # self.LIRS.print_stack()
            # self.LIRS.f3.write("\nHISTORY_1\n")
            # for page in self.Hist1:
            #     self.LIRS.f3.write("page %d\n" % page)


        assert pg in self.LFUalg
        assert pg in self.LIRS
        assert pg in self.LFUalg.PQ
        freq1 = self.LFUalg.PQ.delete(pg)
        freq2 = self.LIRS.delete(pg, poly)

        if self.log:
            self.LIRS.f3.write("\nAfter\n")
            self.LIRS.print_stack()
            self.LIRS.f3.write("\n LFU History:\n")
            for p in self.Hist1:
                self.LIRS.f3.write("page: <%d> freq:<%d>\n" % (p, self.Hist1[p]))
        #assert freq1 == freq2, "page:<%d> f1<%d>, d2<%d>" % (pg, freq1, freq2)
        return freq1


    def getQ(self):
        lamb = 0.05
        return (1 - lamb) * self.W + lamb

    ############################################
    ## Choose a page based on the q distribution
    ############################################
    def chooseRandom(self):
        r = np.random.rand()
        if r < self.W[0]:
            return 0
        return 1

    def addToseparateHistory(self, poly, cacheevict, evicteds_freq):
        histevict = None
        assert len(self.Hist1) <= self.H
        #if (poly == 0) or (poly==-1 and np.random.rand() < 0.5):
        if (poly == 0):
            if len(self.Hist1) == self.H:
                histevict = next(iter(self.Hist1))
                if histevict is not None:
                    del self.Hist1[histevict]
            self.Hist1[cacheevict] = evicteds_freq
            assert len(self.Hist1) >= 1

        if histevict is not None:
            del self.eTime[histevict]


    # def updateSamples(self) :
    #     if self.SampleChangeInHR.full():
    #                 self.SampleCIR -= self.SampleChangeInHR.get()
    #     self.SampleChangeInHR.put(self.NewChangeInHR)
    #     self.SampleCIR += self.NewChangeInHR

    #     if self.SampleHitQ.full():
    #             self.SampleCacheHit -= self.SampleHitQ.get()
    #     self.SampleHitQ.put(self.NewHR)
    #     self.SampleCacheHit += self.NewHR

    def updateInDeltaDirection(self, delta_LR):

        delta = 0
        delta_HR = 1

        if (delta_LR > 0 and self.NewChangeInHR > 0) or (delta_LR < 0 and self.NewChangeInHR < 0):
            delta = 1
        elif (delta_LR < 0 and self.NewChangeInHR > 0) or (delta_LR > 0 and self.NewChangeInHR < 0):
            delta = -1

        elif (delta_LR > 0 or delta_LR < 0) and self.NewChangeInHR == 0:
            delta_HR = 0

        return delta, delta_HR

    def updateInRandomDirection(self):

        if self.learning_rate >= 1:
            self.learning_rate = 0.9
            # print("After LR equal and Inside negative extreme")
        elif self.learning_rate <= 0.001:
            self.learning_rate = 0.005
        else:
            val = round(np.random.uniform(0.001, 0.1), 3)
            val = np.random.rand()
            increase = np.random.choice([True, False])

            if increase:
                self.learning_rate = min(self.learning_rate + abs(self.learning_rate * 0.25), 1)
            else:

                self.learning_rate = max(self.learning_rate - abs(self.learning_rate * 0.25), 0.001)

    def updateLearningRates(self):

        if self.time % (self.seq_len) == 0:

            self.NewHR = round(self.CacheHit / float(self.seq_len), 3)
            self.NewChangeInHR = round(self.NewHR - self.PreviousHR, 3)

            # self.updateSamples()
            delta_LR = round(self.NewLR, 3) - round(self.PreviousLR, 3)
            delta, delta_HR = self.updateInDeltaDirection(delta_LR)

            # if self.page_fault_count >=  self.N:
            #     self.page_fault_count  = 0
            #     if self.W[0] > 0.5 :
            #         self.W = np.array([self.initial_weight,1-self.initial_weight], dtype=np.float32)

            if delta > 0:
                self.learning_rate = min(self.learning_rate + abs(self.learning_rate * delta_LR), 1)

                # print("Inside positive update",delta_LR, self.NewChangeInHR,self.learning_rate)
                self.hitrate_negative_counter = 0
                self.hitrate_zero_counter = 0


            elif delta < 0:
                self.learning_rate = max(self.learning_rate - abs(self.learning_rate * delta_LR), 0.001)
                # print("Inside negative update", delta_LR, self.NewChangeInHR, self.learning_rate)
                self.hitrate_negative_counter = 0
                self.hitrate_zero_counter = 0


            elif delta == 0 and (self.NewChangeInHR <= 0):

                if (self.NewHR <= 0 and self.NewChangeInHR <= 0) or self.NewChangeInHR < 0:
                    self.hitrate_zero_counter += 1

                if self.NewChangeInHR < 0:
                    self.hitrate_negative_counter += 1

                if self.hitrate_zero_counter >= 10:

                    self.learning_rate = self.reset_point
                    self.timer = 0
                    # if self.hitrate_negative_counter >= 5:
                    #     self.hitrate_negative_counter = 0
                    #     if self.W[0] > 0.5:
                    #         self.W = np.array([self.initial_weight,1-self.initial_weight], dtype=np.float32)
                    # if self.W[0] > 0.5:
                    #     self.W = np.array([self.initial_weight,1-self.initial_weight], dtype=np.float32)

                    self.hitrate_zero_counter = 0
                    # self.hitrate_negative_counter = 0

                elif self.NewChangeInHR < 0:

                    # # if self.hitrate_negative_counter >= 5:
                    # #     self.hitrate_negative_counter = 0
                    # if self.W[0] > 0.5:
                    #     self.W = np.array([self.initial_weight, 1 - self.initial_weight], dtype=np.float32)
                    #
                    # else:
                    self.updateInRandomDirection()

            self.PreviousLR = self.NewLR
            self.NewLR = self.learning_rate
            self.PreviousHR = self.NewHR
            self.PreviousChangeInHR = self.NewChangeInHR
            self.CacheHit = 0

    def updateSeparateRewards(self, page):
        reward = np.array([0, 0], dtype=np.float32)

        if page in self.Hist1:
            self.page_fault_count = 0
            pageevict = page
            self.Hist1.delete(page)
            reward[0] = -1
            # if self.W[0]>= 0.5:
            #         reward[0] = - 1.0
            # else:
            #     reward[0] = - 1.0 / (  (self.time-self.eTime[page]) )

            self.info['lru_misses'] += 1

        elif page in self.Hist2:
            self.page_fault_count = 0
            pageevict = page
            self.Hist2.delete(page)
            reward[1] = -1
            # if self.W[1]>= 0.5:
            #         reward[1] = - 1.0
            # else:
            #     reward[1] = - 1.0 / (  (self.time-self.eTime[page]) )

            self.info['lfu_misses'] += 1
        else:
            self.page_fault_count += 1

        return reward

    ########################################################################################################################################
    ####REQUEST#############################################################################################################################
    ########################################################################################################################################
    def request(self, page):
        if self.log:
            self.LIRS.f3.write("_________________________________________________________________________________________\n")
            self.LIRS.f3.write("\n_______________Request <%d>______________\n" % page)
            self.LIRS.print_stack()
            self.LIRS.f3.write("\n LFU History:\n")
            for p in self.Hist1:
                self.LIRS.f3.write("page: <%d> freq:<%d>\n" % (p, self.Hist1[p]))
            # self.LIRS.f3.write("\nLFU arr:\n")
            for v in iter(self.LFUalg.PQ._priorityqueue__freq):
                #self.LIRS.f3.write("page <%d> \n" %v)
                assert v in self.LIRS, "%d" % v
            for v in self.LIRS.lirStack:
                if self.LIRS.lirStack[v].isResident:
                    assert v in self.LFUalg, "%d" % v
            for v in self.LIRS.residentHIRList:
                if v not in self.LFUalg:
                    print("3 missing page %d \n" % v)
                    self.LIRS.print_stack()
                    raise RuntimeError("bad")

        page_fault = False
        self.time = self.time + 1
        self.timer = self.timer + 1

        #####################
        ## Visualization data
        #####################
        if self.Visualization:
            self.X.append(self.time)
            self.Y1.append(self.W[0])
            self.Y2.append(self.W[1])

        #####################################################
        ## Adapt learning rate Here
        ######################################################
        # if self.SAMPLE_W_0Q.full():
        #         self.Sample_W_0 -= self.SAMPLE_W_0Q.get()
        # self.SAMPLE_W_0Q.put(self.W[0])
        # self.Sample_W_0 += self.W[0]
        # self.learning_rate = min (1, max(0.001, self.learning_rate /  np.sqrt(self.timer)))

        self.updateLearningRates()

        ##########################
        ## Process page request
        ##########################
        if page in self.LFUalg:
            assert page in self.LIRS
            if self.log:
                self.LIRS.f3.write("\nHIT\n")
            page_fault = False
            self.pageHitUpdate(page)
            self.CacheHit += 1
        else:
            if self.log:
                self.LIRS.f3.write("\nMISS\n")
            #####################################################
            ## Learning step: If there is a page fault in history
            #####################################################
            pageevict = None

            reward = np.array([0, 0], dtype=np.float32)
            from_lfu_hist = False
            saved_freq = 0
            if page in self.Hist1:
                self.page_fault_count = 0
                pageevict = page
                if (self.Hist1[page] != None):
                   saved_freq = self.Hist1[page]
                from_lfu_hist = True
                del self.Hist1[page]
                reward[0] = -1

            elif page in self.LIRS.hirStack:
                # assert stuff
                assert page not in self.LIRS
                if page in self.LIRS.hirStack:
                    assert page in self.LIRS.lirStack
                    reed  = self.LIRS.lirStack[page].isResident
                    reed2 = self.LIRS.hirStack[page].isResident
                    assert (not reed) and (not reed2)
                ## assert stuff

                self.page_fault_count = 0
                pageevict = page
                saved_freq = self.LIRS.hirStack[page].freq

                #self.Hist2.delete(page)
                reward[1] = -1
            else:
                saved_freq = 0
                self.page_fault_count += 1
            #################
            ## Update Weights
            #################
            if pageevict is not None:
                self.W = self.W * np.exp(self.learning_rate * reward)
                self.W = self.W / np.sum(self.W)

                if self.W[0] >= 0.99:
                    self.W = np.array([0.99, 0.01], dtype=np.float32)

                elif self.W[1] >= 0.99:
                    self.W = np.array([0.01, 0.99], dtype=np.float32)

            ####################
            ## Remove from Cache
            ####################

            #poly = -2
            if self.LFUalg.PQ.size() == self.N: #cache is full
                assert self.LIRS.currentLIRSSize + self.LIRS.currentHIRSSize == self.N
                ################
                ## Choose Policy
                ################
                act = self.chooseRandom()
                cacheevict, poly = self.selectEvictPage(act)
                self.eTime[cacheevict] = self.time

                #################
                ## Remove from Cache and Add to history
                #################
                evicteds_freq = self.evictPage(cacheevict, poly)
                # self.block_lifetime_duration +=  self.time - self.page_lifetime_cache[cacheevict]
                self.unique_block_count += 1
                # self.block_lifetime_durations.append(self.time - self.page_lifetime_cache[cacheevict])
                # print( "Page", cacheevict, "Lifetime", self.page_lifetime_cache[cacheevict],"At time", self.time, "Duration", self.time - self.page_lifetime_cache[cacheevict], "Block in Cache count", self.unique_block_count  )
                # del self.page_lifetime_cache[cacheevict]
                # self.addToHistory(poly, cacheevict)
                self.addToseparateHistory(poly, cacheevict, evicteds_freq)

            self.addToCache(page, saved_freq, from_lfu_hist)
            # self.page_lifetime_cache[page] = self.time
            # print( "Page added to Cache for the first time", page, "Initial Lifetime", self.time  )

            page_fault = True

        ## Count pollution

        if page_fault:
            self.unique_cnt += 1
        self.unique[page] = self.unique_cnt

        # if not page_fault and page in self.page_entering_cache :
        #         self.block_reused_duration +=  self.time - self.page_entering_cache[page]
        #         self.reused_block_count += 1
        #         self.page_entering_cache[page] =  self.time

        # else:
        #     self.page_entering_cache[page] =  self.time

        # if self.time % self.N == 0:
        #     self.pollution = 0
        #     for pg in iter(self.LFUalg.PQ):
        #         if self.unique_cnt - self.unique[pg] >= 2 * self.N:
        #             self.pollution += 1
        #
        #     self.pollution_dat_x.append(self.time)
        #     self.pollution_dat_y.append(100 * self.pollution / self.N)
        # self.pollution_dat_y_val += 100 * self.pollution / self.N
        # self.pollution_dat_y_sum.append(self.pollution_dat_y_val)

        self.learning_rates.append(self.learning_rate)
        return page_fault

    def get_list_labels(self):
        return ['L']

