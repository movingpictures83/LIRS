from page_replacement_algorithm import page_replacement_algorithm
from collections import OrderedDict

class CacheMetaData(object):
    def __init__(self):
        super(CacheMetaData, self).__setattr__("isLir", False)
        super(CacheMetaData, self).__setattr__("isResident", False)
        super(CacheMetaData, self).__setattr__("freq", 0)

    def __setattr__(self, name, value):
        if not hasattr(self, name):
            raise AttributeError("Creating new attributes is not allowed!")
        super(CacheMetaData, self).__setattr__(name, value)


class lirs4_2_1_1(page_replacement_algorithm):

    def __init__(self, cache_size, hirsRatio=0.07):
        self.f3 = open('debugFile', 'w+')
        self.f4 = open("errorFile", "w+")
        self.size = cache_size
        #self.size = int(param['cache_size'])
        if self.size < 4:
             self.size = 4

        print("size : %d" % self.size)
        self.hirsRatio = hirsRatio

        self.hirsSize = int((float(self.size) * self.hirsRatio))
        if self.hirsSize < 2:
            self.hirsSize = 2
        self.maxLirSize = self.size - self.hirsSize
        self.numberHIRpages = 0
        self.numberLIRpages = 0
        self.SStack = OrderedDict()
        self.QStack = OrderedDict()
        self.nonresidentHIRsInS = OrderedDict()
        self.pageFault = True
        self.M = 0.75 # or 1

    def __contains__(self, page):
        if page in self.SStack:
            return self.SStack[page].isResident
        elif page in self.QStack:
            return True
        else:
            return False

    def __getitem__(self, page):
        if page in self.SStack:
            return self.SStack[page]
        elif page in self.QStack:
            return self.QStack[page]
        else:
            raise "Key Error - Not Found %d" % page

    def pageHIT(self, page):
        assert self.numberHIRpages + self.numberLIRpages <= self.size
        assert len(self.QStack) <= self.hirsSize
        assert len(self.SStack) <= (1+self.M) * self.size

        if page in self.SStack:
            if self.SStack[page].isLir:
                self.hitInLir(page)
            else:
                self.hitinHIR(page)
        elif page in self.QStack:
            self.hitInQ(page)
        else:
            raise KeyError("Page not in cache")

    def hitInLir(self, page):
        assert page not in self.nonresidentHIRsInS
        assert page not in self.QStack

        data = self.SStack[page]
        data.freq += 1


        firstKey = next(iter(self.SStack))

        del self.SStack[page]
        self.SStack[page] = data

        if firstKey == page:
            self.pruneS(None)

    def hitinHIR(self, page):
        assert page in self.SStack
        assert page in self.QStack
        assert page not in self.nonresidentHIRsInS
        assert self.SStack[page].freq == self.QStack[page].freq

        firstLir = next(iter(self.SStack))
        firstLirData = self.SStack[firstLir]
        assert firstLirData.isLir == True
        assert firstLirData.isResident == True
        assert firstLir not in self.QStack
        assert firstLir not in self.nonresidentHIRsInS

        del self.QStack[page]
        self.numberHIRpages -= 1

        data = self.SStack[page]
        del self.SStack[page]

        assert self.numberLIRpages <= self.maxLirSize

        data.freq += 1
        data.isLir = True

        assert data.isResident == True

        self.numberLIRpages += 1
        self.SStack[page] = data

        if self.numberLIRpages > self.maxLirSize:
            self.numberLIRpages -= 1
            self.LIRtoHIR()
            self.numberHIRpages += 1
            self.pruneS(None)
            assert self.numberLIRpages == self.maxLirSize

    def LIRtoHIR(self):
        firstLirkey = next(iter(self.SStack))
        firsLirData = self.SStack[firstLirkey]

        assert firsLirData.isLir == True
        assert firsLirData.isResident == True

        del self.SStack[firstLirkey]

        assert len(self.QStack) < self.hirsSize
        assert self.numberHIRpages < self.hirsSize
        firsLirData.isLir = False
        self.QStack[firstLirkey] = firsLirData

    def hitInQ(self, page):
        assert page not in self.nonresidentHIRsInS
        assert self.QStack[page].isLir == False
        assert self.QStack[page].isResident == True

        data = self.QStack[page]
        data.freq += 1

        del self.QStack[page]

        self.QStack[page] = data
        self.SStack[page] = data

    def addToCache(self, page, from_lfu, saved_freq):
        assert self.numberHIRpages + self.numberLIRpages <= self.size
        assert self.numberHIRpages <= self.hirsSize
        assert len(self.QStack) == self.numberHIRpages
        assert len(self.SStack) <= (1+self.M) * self.size
        assert self.numberLIRpages <= self.maxLirSize
        assert (self.numberLIRpages < self.maxLirSize) or (self.numberHIRpages < self.hirsSize)
        assert page not in self

        if (self.numberLIRpages < self.maxLirSize) and (self.numberHIRpages < self.hirsSize):
            assert self.numberHIRpages == 0

        if from_lfu:
            print(saved_freq)
            self.promotionFromLFU(page, saved_freq)
        else:
            if page in self.nonresidentHIRsInS:
                assert page in self.SStack

            if page in self.SStack:
                data = self.SStack[page]
                assert data.isLir == False
                assert data.isResident == False
                assert page in self.nonresidentHIRsInS
                assert self.nonresidentHIRsInS[page] is self.SStack[page]

                self.hitInHistory(page)
            else:
                assert page not in self.QStack
                self.fullMiss(page)

    def hitInHistory(self, page):
        if self.numberLIRpages == self.maxLirSize:
            assert self.numberHIRpages < self.hirsSize
            assert len(self.QStack) < self.hirsSize
            assert page not in self.QStack

            data = self.SStack[page]

            del self.SStack[page]
            del self.nonresidentHIRsInS[page]

            data.freq += 1
            data.isLir = True

            assert data.isResident == False
            data.isResident = True

            self.SStack[page] = data

            self.LIRtoHIR()
            self.numberHIRpages += 1
            self.pruneS(None)
        else:
            assert page not in self.QStack
            data = self.SStack[page]
            del self.SStack[page]
            del self.nonresidentHIRsInS[page]

            self.numberLIRpages += 1
            data.freq += 1
            data.isLir = True

            assert data.isResident == False
            data.isResident = True

            self.SStack[page] = data

    def fullMiss(self, page):
        assert self.numberHIRpages == len(self.QStack)
        data = CacheMetaData()
        data.isResident = True
        data.freq = 1

        if self.numberLIRpages == self.maxLirSize:
            data.isLir = False
            self.SStack[page] = data
            self.QStack[page] = data
            self.numberHIRpages += 1
        else:
            if self.numberHIRpages == 0:
                data.isLir = True
                self.SStack[page] = data
                self.numberLIRpages += 1
            else:
                #Arbitrary Case
                if self.numberHIRpages == self.hirsSize:
                    lastHirKey = next(iter(self.QStack))
                    lastHirData = self.QStack[lastHirKey]

                    assert lastHirKey not in self.nonresidentHIRsInS
                    assert lastHirData.isLir == False
                    assert lastHirData.isResident == True

                    if lastHirKey in self.SStack:
                        self.forceHIRtoLIR()
                        data.isLir = False
                        self.SStack[page] = data
                        self.QStack[page] = data
                        self.numberHIRpages += 1
                    else:
                        self.putAtTheBottonOfS(page)
                else:
                    raise Exception("Should not happen")
                    # bruh
    def forceHIRtoLIR(self):
        lastHirKey = next(iter(self.QStack))
        lastHirData = self.QStack[lastHirKey]

        assert lastHirData.isLir == False
        assert lastHirData.isResident == True
        assert lastHirKey not in self.nonresidentHIRsInS
        del self.QStack[lastHirKey]

        lastHirData.isLir = True
        lastHirData.isResident = True

        assert self.SStack[lastHirKey] is lastHirData

        if lastHirKey in self.SStack:
            del self.SStack[lastHirKey]
        self.ordered_dict_prepend(self.SStack, lastHirKey, lastHirData)
        ## Or not prepend! choices choices!

        self.numberLIRpages += 1
        self.numberHIRpages -= 1

    def putAtTheBottonOfS(self, page):
        data = CacheMetaData()
        data.isResident = True
        data.freq = 1
        data.isLir = True

        self.ordered_dict_prepend(self.SStack, page, data)
        self.numberLIRpages += 1



    def promotionFromLFU(self, page, saved_freq):
        assert page not in self.nonresidentHIRsInS

        data = CacheMetaData()
        data.freq = saved_freq + 1
        data.isLir = True
        data.isResident = True

        if self.numberLIRpages == self.maxLirSize:
            assert self.numberHIRpages == len(self.QStack)
            assert len(self.QStack) < self.hirsSize
            assert page not in self.QStack
            assert page not in self.SStack

            self.SStack[page] = data

            self.LIRtoHIR()
            self.numberHIRpages += 1
            self.pruneS(None)
        else:
            assert self.numberHIRpages == 0 or self.numberHIRpages == self.hirsSize

            self.SStack[page] = data
            self.numberLIRpages += 1
            # bruh
            pass

    def delete(self, page, lfu_deletion, avoid_removal):
        assert self.numberLIRpages == self.maxLirSize and self.numberHIRpages == self.hirsSize
        assert self.numberHIRpages + self.numberLIRpages == self.size
        assert len(self.SStack) <= (1+self.M) * self.size
        assert page in self
        assert page not in self.nonresidentHIRsInS

        if lfu_deletion:
            if page in self.SStack:
                if self.SStack[page].isLir:
                    return self.deleteLirPageByLFU(page, avoid_removal)
                else:
                    return self.deleteHirPageByLFU(page)
            elif page in self.QStack:
                return self.deleleteHirQByLFU(page)
            else:
                raise Exception("Page not in self - LFU")
        else:
            firstQkey = next(iter(self.QStack))
            assert firstQkey == page

            if page in self.SStack:
                assert self.SStack[page].isLir == False
                return self.deleteHirInS(page, avoid_removal)
            elif page in self.QStack:
                return self.deleteHirInQ(page)
            else:
                raise Exception("Page not in self - LIRS")

    def deleteLirPageByLFU(self, page, avoid_removal):
        data = self.SStack[page]
        freq = data.freq

        self.numberLIRpages -= 1

        firstKey = next(iter(self.SStack))
        firstData = self.SStack[firstKey]

        assert firstData.isLir == True
        assert firstData.isResident == True
        assert firstKey not in self.QStack
        assert firstKey not in self.nonresidentHIRsInS


        del self.SStack[page]

        if firstKey == page:
            # Bro be careful with the pruns
            self.pruneS(avoid_removal)

        assert freq > 0
        return freq

    def deleteHirPageByLFU(self, page):
        assert page in self.QStack

        data = self.SStack[page]

        assert data.isLir == False
        assert data.isResident ==True

        freq = data.freq

        assert self.QStack[page].freq == freq
        assert freq > 0

        del self.SStack[page]
        del self.QStack[page]

        self.numberHIRpages -= 1

        return freq

    def deleleteHirQByLFU(self, page):
        data = self.QStack[page]

        assert data.isLir == False
        assert data.isResident == True

        freq = data.freq
        assert freq > 0

        del self.QStack[page]
        self.numberHIRpages -= 1

        return freq

    def deleteHirInS(self, page, avoid_removal):
        data = self.SStack[page]

        assert data.isLir == False
        assert data.isResident == True
        assert page in self.QStack


        del self.QStack[page]
        self.numberHIRpages -= 1

        data.isResident = False

        assert self.SStack[page].isResident == False

        self.nonresidentHIRsInS[page] = data

        if len(self.nonresidentHIRsInS) > int(float(self.size) * self.M):
            iterObj = iter(self.nonresidentHIRsInS)
            firstNonRed = next(iterObj)

            #(bro be careful with  the deletes)
            if firstNonRed == avoid_removal:
                firstNonRed =  next(iterObj)
            del self.nonresidentHIRsInS[firstNonRed]
            del self.SStack[firstNonRed]

            assert len(self.nonresidentHIRsInS) == int(float(self.size) * self.M)

        freq = data.freq

        assert freq > 0

        return freq

    def deleteHirInQ(self, page):
        data = self.QStack[page]

        assert data.isLir == False
        assert data.isResident == True
        assert page not in self.nonresidentHIRsInS

        del self.QStack[page]
        self.numberHIRpages -= 1

        freq = data.freq

        assert freq > 0

        return freq

    def request(self, page):
        self.f3.write("\nRequest: "+str(page)+"\n")
        self.print_stack()
        if page in self:
            self.f3.write("HIT\n")
            self.pageHIT(page)
        else:
            self.f3.write("MISS\n")
            if self.numberHIRpages + self.numberLIRpages == self.size:
                evictPage = next(iter(self.QStack))
                self.f3.write("Full evict: "+str(evictPage))
                #self.f3.write("Full evict %d\n" % evictPage)
                self.delete(evictPage, False, page)
            else:
                self.f3.write("Not full\n")
            self.addToCache(page, False, -1)

    def pruneS(self, avoid_removal):
        pruneKeys = []
        for key in self.SStack:
            data = self.SStack[key]

            if data.isLir:
                break

            #assert data.isResident == False

            if key == avoid_removal:
                avoid_removal_data = self.SStack[avoid_removal]
                del self.SStack[avoid_removal]
                self.SStack[avoid_removal] = avoid_removal_data
                continue
            #del self.SStack[key]
            pruneKeys.append(key)
            if key in self.nonresidentHIRsInS:
                del self.nonresidentHIRsInS[key]
            else:
                assert key in self.QStack
        for key in pruneKeys:
            del self.SStack[key]

    def print_stack(self):
        Lirs5 = self
        Lirs5.f3.write("Stats:\n")
        Lirs5.f3.write("LIR pages: %d\%d\n" % (self.numberLIRpages, self.maxLirSize))
        Lirs5.f3.write("Q pages: %d len: %d\%d\n" % (self.numberHIRpages, len(self.QStack), self.hirsSize))
        Lirs5.f3.write("NonRed pages: %d\%d\n" % (len(self.nonresidentHIRsInS), int(float(self.size)*self.M)))
        count_lir = 0
        for key in self.SStack:
            data = self.SStack[key]
            Lirs5.f3.write("S "+str(key)+" red: "+str(data.isResident)+" lir: "+str(data.isLir)+"\n")
            if data.isLir:
                count_lir += 1

        Lirs5.f3.write("\n\nQ stack:\n")
        for key in self.QStack:
            data = self.QStack[key]
            Lirs5.f3.write("Q "+str(key)+" red: "+str(data.isResident)+" lir: "+str(data.isLir)+"\n")

        Lirs5.f3.write("\n\nresidentHIRList stack:\n")
        for key in self.nonresidentHIRsInS:
            data = self.nonresidentHIRsInS[key]
            Lirs5.f3.write("nonRed "+str(key)+" red: "+str(data.isResident)+" lir: "+str(data.isLir)+"\n")
        Lirs5.f3.write("\n\n")

        assert count_lir == self.numberLIRpages

    def check(self):
        count_lir = 0
        for v in self.SStack:
            data = self.SStack[v]
            assert data.freq > 0

            if data.isLir:
                count_lir += 1
                assert data.isResident == True
                assert v not in self.QStack
                assert v not in self.nonresidentHIRsInS
            elif data.isResident:
                assert v in self.QStack
                assert v not in self.nonresidentHIRsInS
                assert data is self.QStack[v]
            else:
                assert v in self.nonresidentHIRsInS
                assert data is self.nonresidentHIRsInS[v]
        assert count_lir == self.numberLIRpages
        assert len(self.QStack) == self.numberHIRpages

        for v in self.QStack:
            data = self.QStack[v]
            assert data.isResident == True
            assert data.isLir == False
            assert data.freq > 0
            assert v not in self.nonresidentHIRsInS

            if v in self.SStack:
                assert data is self.SStack[v]

        for v in self.nonresidentHIRsInS:
            data = self.nonresidentHIRsInS[v]
            assert data.isResident == False
            assert data.isLir == False
            assert data.freq > 0
            assert v in self.SStack
            assert v not in self.QStack


    def get_block_reused_duration(self):
        return 0


    def ordered_dict_prepend(self, dct, key, value, dict_setitem=dict.__setitem__):
        dct[key] = value
        dct.move_to_end(key, last=False)
        #root = dct.__root
        #first = root[1]

        #if key in dct:
        #    link = dct._OrderedDict__map[key]
        #    link_prev, link_next, _ = link
        #    link_prev[1] = link_next
        #    link_next[0] = link_prev
        #    link[0] = root
        #    link[1] = first
        #    root[1] = first[0] = link
        #else:
        #    root[1] = first[0] = dct._OrderedDict__map[key] = [root, first, key]
        #    dict_setitem(dct, key, value)



    def __del__(self):
        self.f3.close()
        print('Closing file')


if __name__ == "__main__":
    params = {'cache_size': 6}
