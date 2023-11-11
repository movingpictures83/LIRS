from page_replacement_algorithm import page_replacement_algorithm
from collections import OrderedDict

class CacheMetaData:
    def __init__(self):
        self.isLir = False
        self.isResident = False


class Lirs2(page_replacement_algorithm):
    f3 = open('debugFile', 'w+')
    def __init__(self, cache_size):

        self.size = cache_size
        if self.size < 10:
             self.size = 10
        print("size : %d" % self.size)
        self.hirsRatio = 0.1
        self.hirsSize = int((float(self.size) * self.hirsRatio + 0.5))
        if self.hirsSize < 2:
            self.hirsSize = 2
        self.lirsSize = self.size - self.hirsSize
        self.currentHIRSSize = 0
        self.currentLIRSSize = 0
        self.lirStack = OrderedDict()
        self.hirStack = OrderedDict()
        self.residentHIRList = OrderedDict()
        self.nonresidentHIRsInStack = 0
        self.numSteps = 0
        self.pageFault = True

    def __contains__(self, page):
        if page in self.lirStack:
            return self.lirStack[page].isResident
        elif page in self.residentHIRList:
            return True
        else:
            return False

    def request(self, page):
        #Lirs2.f3.write("_____Request %d _____\n" % page)
        #Lirs2.f3.write("size: lirs: %d hir: %d non_red: %d\n" % (
        #self.currentLIRSSize, self.currentHIRSSize, self.nonresidentHIRsInStack))
        assert self.currentHIRSSize + self.currentLIRSSize <= self.size + 1
        self.pageFault = False
        if page in self.lirStack:
            if self.lirStack[page].isLir:
                self.hitInLIRS(page)
            else:
                self.hitInHIRInLIRStack(page)
        elif page in self.residentHIRList:
            self.hitInHIRList(page)
        else:
            self.processMiss(page)
            self.pageFault = True

        #self.print_stack()
        return self.pageFault


    def hitInHIRList(self, page):
        data = self.residentHIRList[page]
        del self.residentHIRList[page]

        self.residentHIRList[page] = data
        self.lirStack[page] = data
        self.hirStack[page] = data

        self.limitStackSize()

    def hitInLIRS(self, page):
        firstKey = next(iter(self.lirStack))
        data =  self.lirStack[page]
        del self.lirStack[page]
        self.lirStack[page] = data
        if firstKey == page:
            self.pruneStack()

    def hitInHIRInLIRStack(self, page):
        data = self.lirStack[page]
        result = data.isResident
        data.isLir = True
        del self.lirStack[page]
        del self.hirStack[page]
        if result:
            del self.residentHIRList[page]
            self.currentHIRSSize -= 1
        else:
            data.isResident = True
            self.nonresidentHIRsInStack -= 1

        if self.currentLIRSSize >= self.lirsSize:
            self.ejectLIR()

        self.lirStack[page] = data
        self.currentLIRSSize += 1

        return result

    def ejectLIR(self):
        firstKey = next(iter(self.lirStack))
        tmpData = self.lirStack[firstKey]
        tmpData.isLir = False
        del self.lirStack[firstKey]

        self.currentLIRSSize -= 1
        if self.currentHIRSSize >= self.hirsSize:
            self.ejectResidentHIR()
        self.residentHIRList[firstKey] = tmpData
        self.currentHIRSSize += 1
        self.pruneStack()

    def ejectResidentHIR(self):
        firstKey = next(iter(self.residentHIRList))
        del self.residentHIRList[firstKey]
        if firstKey in self.lirStack:
            tmpData = self.lirStack[firstKey]
            tmpData.isResident = False
            self.nonresidentHIRsInStack += 1
        self.currentHIRSSize -= 1



    def limitStackSize(self):
        pruneSize = self.currentHIRSSize + self.currentLIRSSize + self.nonresidentHIRsInStack - self.size * 2

        pruneKeys = []
        for key in self.hirStack:
            data = self.hirStack[key]
            self.numSteps += 1
            if pruneSize <= 0:
                break
            pruneKeys.append(key)
            #del self.lirStack[key]
            #del self.hirStack[key]

            if not data.isResident:
                self.nonresidentHIRsInStack -= 1
            pruneSize -= 1
        for key in pruneKeys:
            del self.lirStack[key]
            del self.hirStack[key]

    def pruneStack(self):
        lirKeys = []
        hirKeys = []
        for key in self.lirStack:
            data = self.lirStack[key]
            self.numSteps += 1
            if data.isLir:
                break
            #del self.lirStack[key]

            lirKeys.append(key)
            if key in self.hirStack:
                hirKeys.append(key)
                #del self.hirStack[key]
            else:
                assert key in self.residentHIRList, "key is %d" % key

            if not data.isResident:
                self.nonresidentHIRsInStack-= 1
        for key in lirKeys:
            del self.lirStack[key]
        for key in hirKeys:
            del self.hirStack[key]


    def processMiss(self, page):
        if self.currentLIRSSize < self.lirsSize and self.currentHIRSSize == 0:
            data = CacheMetaData()
            data.isLir = True
            data.isResident = True
            self.lirStack[page] =  data
            self.currentLIRSSize += 1
            return
        elif self.currentHIRSSize >= self.hirsSize:
            self.ejectResidentHIR()

        data = CacheMetaData()
        data.isLir = False
        data.isResident = True

        self.lirStack[page] =  data
        self.hirStack[page] = data
        self.residentHIRList[page] = data

        self.currentHIRSSize += 1
        self.limitStackSize()

    def delete(self, page):
        #Lirs2.f3.write("_____Delete %d _____\n" % page)
        #Lirs2.f3.write("size: lirs: %d hir: %d non_red: %d\n" % (self.currentLIRSSize, self.currentHIRSSize, self.nonresidentHIRsInStack))
        if page in self.lirStack:
            if self.lirStack[page].isLir:
                #Lirs2.f3.write("delete lir\n")
                self.deleteLIRpage(page)
            else:
                #Lirs2.f3.write("delete hir in lirs\n")
                self.deleteHIRInLIRStack(page)
        elif page in self.residentHIRList:
            #Lirs2.f3.write("delete red\n")
            self.deleteResidentHIR(page)
        else:
            raise KeyError("Page not in cache")
        #Lirs2.f3.write("print stack: \n")
        #self.print_stack()

    def deleteResidentHIR(self, page):
        del self.residentHIRList[page]
        if page in self.lirStack:
            tmpData = self.lirStack[page]
            tmpData.isResident = False
            self.nonresidentHIRsInStack += 1
        self.currentHIRSSize -= 1

    def deleteLIRpage(self, page):
        del self.lirStack[page]  # NBW

        if self.currentHIRSSize > 1:
            assert self.currentLIRSSize == self.lirsSize
            self.forceHIRtoLIR()

        self.currentLIRSSize -= 1

        # firstKey = next(iter(self.lirStack))
        self.pruneStack()
        # if firstKey == page:
        #     #Lirs2.f3.write("Stack before prune")
        #     #self.print_stack()



    def deleteHIRInLIRStack(self, page):
        data = self.lirStack[page]
        result = data.isResident

        if result:
            del self.residentHIRList[page]
            self.currentHIRSSize -= 1
        else:
            raise KeyError("Page not in cache")

        data.isResident = False

        self.nonresidentHIRsInStack += 1


    def forceHIRtoLIR(self):
        firstKey = next(iter(self.residentHIRList))
        firstHIRdata = self.residentHIRList[firstKey]

        #Lirs2.f3.write("\nFirstkey: %d\n" % firstKey)

        del self.residentHIRList[firstKey]

        firstHIRdata.isLIR = True
        firstHIRdata.isReisident = True

        if firstKey in self.lirStack:
            assert firstKey in self.hirStack

            del self.hirStack[firstKey]

            #self.lirStack[firstKey] = firstHIRdata
            self.lirStack[firstKey].isLir = True  # Q1
            self.currentLIRSSize += 1
        else:
            self.ordered_dict_prepend(self.lirStack, firstKey, firstHIRdata)
            self.lirStack[firstKey].isLir = True
            self.lirStack[firstKey].isReisident = True
            self.currentLIRSSize += 1

        #
        # if firstKey in self.hirStack:
        #     del self.hirStack[firstKey]
        #
        # if firstKey in self.lirStack:
        #     self.lirStack[firstKey].isLir = True  #Q1
        #     self.currentLIRSSize += 1
        # else:
        #     self.ordered_dict_prepend(self.lirStack, firstKey, firstHIRdata)


        #Lirs2.f3.write("\nis LIR %r\n" % self.lirStack[firstKey].isLIR)

        self.currentHIRSSize -= 1

    def print_stack(self):
        Lirs2.f3.write("LIR stack:")
        for key in self.lirStack:
            data = self.lirStack[key]
            Lirs2.f3.write("LIR <%d> red: <%r> lir: <%r>\n" % (key, data.isResident, data.isLir))

        Lirs2.f3.write("\n\nHIR stack:")
        for key in self.hirStack:
            data = self.hirStack[key]
            Lirs2.f3.write("HIR <%d> red: <%r> lir: <%r>\n" % (key, data.isResident, data.isLir))

        Lirs2.f3.write("\n\nresidentHIRList stack:")
        for key in self.residentHIRList:
            data = self.residentHIRList[key]
            Lirs2.f3.write("redHIR <%d> red: <%r> lir: <%r>\n" % (key, data.isResident, data.isLir))
        Lirs2.f3.write("\n\n")

    def get_block_reused_duration(self):
        return 0

    def ordered_dict_prepend(self, dct, key, value, dict_setitem=dict.__setitem__):
        root = dct._OrderedDict__root
        first = root[1]

        if key in dct:
            link = dct._OrderedDict__map[key]
            link_prev, link_next, _ = link
            link_prev[1] = link_next
            link_next[0] = link_prev
            link[0] = root
            link[1] = first
            root[1] = first[0] = link
        else:
            root[1] = first[0] = dct._OrderedDict__map[key] = [root, first, key]
            dict_setitem(dct, key, value)


    def __del__(self):
        Lirs2.f3.close()
        print('Closing file')


if __name__ == "__main__":
    params = {'cache_size': 6}
    alg = Lirs2(params)

    # alg.request(1)
    # alg.request(2)
    # alg.request(3)
    # alg.request(4)
    # alg.print_stack()
    # alg.delete(2)
    # alg.print_stack()
    # alg.request(5)
    # alg.delete(1)
    # alg.print_stack()
    #
    # alg.request(3)
    # alg.print_stack()
    # alg.request(6)
    # alg.request(7)
    # alg.print_stack()
    # alg.delete(7)
    # alg.print_stack()
    # alg.request(8)
    # alg.print_stack()
    # alg.request(9)
    # alg.print_stack()
    # alg.delete(9)
    # alg.print_stack()
    # alg.request(4)
    # alg.request(5)
    # alg.request(3)
    # alg.request(6)
    # alg.print_stack()
    # alg.delete(8)
    # alg.print_stack()



    # total_pg_refs = 0
    # num_pg_fl = 0
    #
    # f = open('m.txt', 'r')
    # last_ref_block = -1
    # for line in f:
    #     try:
    #         ref_block = int(line)
    #     except:
    #         continue
    #     print("________Adding %d _________" % ref_block)
    #     total_pg_refs += 1
    #
    #     pg_fl = alg.request(ref_block)
    #     if(pg_fl):
    #         print("miss")
    #     else:
    #         print("hit")
    #
    #     alg.print_stack()
    #
    #     if pg_fl:
    #         num_pg_fl += 1
    # alg.print_stack()
    # print(total_pg_refs)
    # print(num_pg_fl)
    # print(1.0 - num_pg_fl / total_pg_refs)
