from page_replacement_algorithm import page_replacement_algorithm

class page_n:
    IN_STACK_S = 1
    OUT_STACK_S = 0

    def __init__(self, page):
        self.page = page
        self.isResident = False
        self.isHIR_block = True

        self.LIRS_next = None
        self.LIRS_prev = None

        self.HIR_rsd_next = None
        self.HIR_rsd_prev = None

        self.HIR_rsd_next = None
        self.HIR_rsd_prev = None

        self.HIR_n_rsd_next = None
        self.HIR_n_rsd_prev = None

        self.recency = page_n.OUT_STACK_S


class Lirs(page_replacement_algorithm):
    f3 = open('debugFile', 'w+')
    def __init__(self, cache_size):
        # assert "max_s_len" in param
        # assert "mem_size" in param
        # assert "hir_rate" in param

        # params = {'max_s_len': 300*2500, 'mem_size': 300, 'hir_rate': 1.0}

        self.N = int(cache_size)#param['cache_size'])
        if self.N < 10:
            self.N = 10
        print("Cache Size")
        print(self.N)
        self.MEM_SIZE = self.N
        self.MAX_S_LEN = self.N * 2

        self.HIR_RATE = 1.0

        self.total_pg_refs = 0
        self.num_pg_fl = 0
        self.pg_fl = True

        self.last_ref_block = -1

        self.LRU_list_head = None
        self.LRU_list_tail = None

        self.HIR_list_head = None
        self.HIR_list_tail = None

        self.HIR_n_red_head = None
        self.HIR_n_red_tail = None

        self.LIR_LRU_block = None

        self.HIR_block_portion_limit = int(((self.HIR_RATE / 100.0) * self.MEM_SIZE))

        if self.HIR_block_portion_limit < 2:
            self.HIR_block_portion_limit = 2
        print("hir limit: ")
        print(self.HIR_block_portion_limit)


        self.page_tbl = {}

        self.free_mem_size = self.MEM_SIZE
        self.cur_lir_S_len = 0
        self.num_LIR_pgs = 0

        self.evict_list = []

    def __contains__(self, q):
        if q not in self.page_tbl:
            return False
        return self.page_tbl[q].isResident

    def request(self, page):
        #self.time = self.time + 1

        ref_block = int(page)
        self.total_pg_refs += 1

        self.pg_fl = False

        if ref_block != self.last_ref_block:
            self.last_ref_block = ref_block

            self.pg_fl = self.pageRequest(ref_block)

            if self.pg_fl:
                self.num_pg_fl += 1

        return self.pg_fl

    def pageRequest(self, page):
        if page not in self.page_tbl:
            self.page_tbl[page] = page_n(page)

        cache_miss = False
        page_ref = self.page_tbl[page]

        if not page_ref.isResident:  # cache miss
            if self.free_mem_size == 0:  # cache is full
                self.HIR_list_tail.isResident = False
                self.evict_list.append(self.HIR_list_tail.page)
                self.add_HIR_n_red(self.HIR_list_tail)
                self.remove_HIR_list(self.HIR_list_tail)
                self.free_mem_size += 1
            elif self.free_mem_size > self.HIR_block_portion_limit:
                page_ref.isHIR_block = False
                self.num_LIR_pgs += 1
            self.free_mem_size -= 1
            cache_miss = True
        elif page_ref.isHIR_block:
            self.remove_HIR_list(page_ref)

        self.remove_HIR_n_red(page_ref)
        self.remove_LIRS_list(page_ref)
        # place newly referenced page at head
        self.add_LRU_list_head(page_ref)
        page_ref.isResident = True

        if page_ref.recency == page_n.OUT_STACK_S:
            self.cur_lir_S_len += 1

        if page_ref.isHIR_block and page_ref.recency == page_n.IN_STACK_S:
            page_ref.isHIR_block = False
            self.num_LIR_pgs += 1

            if self.num_LIR_pgs > self.MEM_SIZE - self.HIR_block_portion_limit:
                self.add_HIR_list_head(self.LIR_LRU_block)
                self.HIR_list_head.isHIR_block = True
                self.HIR_list_head.recency = page_n.OUT_STACK_S
                self.num_LIR_pgs -= 1

                self.LIR_LRU_block.recency = page_n.OUT_STACK_S
                self.cur_lir_S_len -= 1
                self.LIR_LRU_block = self.LIR_LRU_block.LIRS_prev

                self.LIR_LRU_block = self.find_last_LIR_LRU()
            else:
                print("Warnning!")
        elif page_ref.isHIR_block:
            self.add_HIR_list_head(page_ref)
        page_ref.recency = page_n.IN_STACK_S
        self.prune_LIRS_stack()

        return cache_miss

    def remove_HIR_n_red(self, page_ref):
        if page_ref is None:
            return False

        if self.HIR_n_red_head is self.HIR_n_red_tail and page_ref is self.HIR_n_red_tail:
            self.HIR_n_red_tail = None
            self.HIR_n_red_head = None
            return True

        if page_ref.HIR_n_rsd_next is None and page_ref.HIR_n_rsd_prev is None:
            return True

        if page_ref.HIR_n_rsd_prev is None:
            self.HIR_n_red_head = page_ref.HIR_n_rsd_next
        else:
            page_ref.HIR_n_rsd_prev.HIR_n_rsd_next = page_ref.HIR_n_rsd_next

        if page_ref.HIR_n_rsd_next is None:
            self.HIR_n_red_tail = page_ref.HIR_n_rsd_prev
        else:
            page_ref.HIR_n_rsd_next.HIR_n_rsd_prev = page_ref.HIR_n_rsd_prev

        page_ref.HIR_n_rsd_prev = None
        page_ref.HIR_n_rsd_next = None

        return True

    def add_HIR_n_red(self, page_ref):
        if page_ref.recency == page_n.OUT_STACK_S:
            return False

        page_ref.HIR_n_rsd_next = self.HIR_n_red_head

        if self.HIR_n_red_head is None:
            self.HIR_n_red_head = page_ref
            self.HIR_n_red_tail = page_ref
        else:
            self.HIR_n_red_head.HIR_n_rsd_prev = page_ref
        self.HIR_n_red_head = page_ref
        return True

    def remove_LIRS_list(self, page_entry):
        if page_entry is None:
            return False
        if page_entry.LIRS_next is None and page_entry.LIRS_prev is None:
            return True

        if page_entry is self.LIR_LRU_block:
            self.LIR_LRU_block = page_entry.LIRS_prev
            self.LIR_LRU_block = self.find_last_LIR_LRU()

        if page_entry.LIRS_prev is None:
            self.LRU_list_head = page_entry.LIRS_next
        else:
            page_entry.LIRS_prev.LIRS_next = page_entry.LIRS_next

        if page_entry.LIRS_next is None:
            self.LRU_list_tail = page_entry.LIRS_prev
        else:
            page_entry.LIRS_next.LIRS_prev = page_entry.LIRS_prev

        page_entry.LIRS_prev = None
        page_entry.LIRS_next = None

        return True

    def find_last_LIR_LRU(self):
        assert self.LIR_LRU_block is not None, "LIR stack is empty"
        while self.LIR_LRU_block.isHIR_block:
            self.LIR_LRU_block.recency = page_n.OUT_STACK_S
            self.cur_lir_S_len -= 1

            if self.LIR_LRU_block is self.HIR_n_red_tail:

                temp_r = self.HIR_n_red_tail
                self.HIR_n_red_tail = self.HIR_n_red_tail.HIR_n_rsd_prev
                if self.HIR_n_red_tail is None:
                    self.HIR_n_red_head = None
                self.remove_HIR_n_red(temp_r)

            self.LIR_LRU_block = self.LIR_LRU_block.LIRS_prev
        return self.LIR_LRU_block

    def remove_HIR_list(self, page_entry):
        if page_entry is None:
            return False

        if page_entry.HIR_rsd_prev is None:
            self.HIR_list_head = page_entry.HIR_rsd_next
        else:
            page_entry.HIR_rsd_prev.HIR_rsd_next = page_entry.HIR_rsd_next

        if page_entry.HIR_rsd_next is None:
            self.HIR_list_tail = page_entry.HIR_rsd_prev
        else:
            page_entry.HIR_rsd_next.HIR_rsd_prev = page_entry.HIR_rsd_prev

        page_entry.HIR_rsd_prev = None
        page_entry.HIR_rsd_next = None

        return True

    #HIR page at the botton will always be nonresident
    def prune_LIRS_stack(self):
        if self.cur_lir_S_len <= self.MAX_S_LEN:
            return None


        tmp_prt = self.HIR_n_red_tail
        self.HIR_n_red_tail = self.HIR_n_red_tail.HIR_n_rsd_prev
        if self.HIR_n_red_tail is None:
            self.HIR_n_red_head = None
        self.remove_HIR_n_red(tmp_prt)

        tmp_prt.recency = page_n.OUT_STACK_S
        self.remove_LIRS_list(tmp_prt)
        self.insert_LRU_list(tmp_prt, self.LIR_LRU_block)
        self.cur_lir_S_len -= 1

        return tmp_prt

    # Doing O(n) prune for now because deleting a random LIRS,
    # causes the HIR non resident pages to be out of order.
    # def prune_LIRS_stack(self):
    #     if self.cur_lir_S_len <= self.MAX_S_LEN:
    #         return None
    #     tmp_prt = self.LIR_LRU_block
    #     while not tmp_prt.isHIR_block:  # hmm
    #         tmp_prt = tmp_prt.LIRS_prev
    #     tmp_prt.recency = page_n.OUT_STACK_S
    #     self.remove_LIRS_list(tmp_prt)
    #     self.insert_LRU_list(tmp_prt, self.LIR_LRU_block)
    #     self.cur_lir_S_len -= 1
    #
    #     return tmp_prt

    def insert_LRU_list(self, old_ref, new_ref):
        old_ref.LIRS_next = new_ref.LIRS_next
        old_ref.LIRS_prev = new_ref

        if new_ref.LIRS_next is not None:
            new_ref.LIRS_next.LIRS_prev = old_ref
        new_ref.LIRS_next = old_ref

    # put a HIR resident block on the end of HIR resident list
    def add_HIR_list_head(self, new_rsd_HIR_ref):
        new_rsd_HIR_ref.HIR_rsd_next = self.HIR_list_head

        if self.HIR_list_head is None:
            self.HIR_list_head = new_rsd_HIR_ref
            self.HIR_list_tail = new_rsd_HIR_ref
        else:
            self.HIR_list_head.HIR_rsd_prev = new_rsd_HIR_ref
        self.HIR_list_head = new_rsd_HIR_ref

    def add_LRU_list_head(self, page_ref):
        page_ref.LIRS_next = self.LRU_list_head

        if self.LRU_list_head is None:
            self.LRU_list_tail = page_ref
            self.LRU_list_head = page_ref
            self.LIR_LRU_block = self.LRU_list_tail  # *since now the point to lir page with Smax isn't nil
        else:
            self.LRU_list_head.LIRS_prev = page_ref
            self.LRU_list_head = page_ref

    def print_stack(self):
        f1 = open('./testfile', 'w+')

        for i in range(len(self.evict_list)):
            f1.write("<%d> %d\n" % (i, self.evict_list[i]))
        f1.close()

    def printS(self):
        Lirs.f3.write("S is: \n")
        tempP = self.LIR_LRU_block
        while tempP is not None:
            Lirs.f3.write("<%d> red: <%r> hir: <%r>\n" % (tempP.page, tempP.isResident, tempP.isHIR_block))
            tempP = tempP.LIRS_prev
        Lirs.f3.write("_____\n")

        Lirs.f3.write("P is: \n")
        tempP = self.HIR_n_red_tail
        while tempP is not None:
            Lirs.f3.write("<%d> red: <%r> hir: <%r>\n" % (tempP.page, tempP.isResident, tempP.isHIR_block))
            tempP = tempP.HIR_n_rsd_prev
        Lirs.f3.write("_____")

        Lirs.f3.write("Q is: \n")
        tempP = self.HIR_list_tail
        while tempP is not None:
            Lirs.f3.write("<%d> red: <%r> hir: <%r>\n" % (tempP.page, tempP.isResident, tempP.isHIR_block))
            tempP = tempP.HIR_rsd_prev
        Lirs.f3.write("_____\n")

    def get_block_reused_duration(self):
        return 0


    def __del__(self):
        Lirs.f3.close()
        print('Closing file')


if __name__ == "__main__":
    params = {'cache_size': 32, 'max_s_len': 10 * 2, 'mem_size': 10, 'hir_rate': 2.0}
    alg = Lirs(params)
    total_pg_refs = 0
    num_pg_fl = 0

    f = open('m.txt', 'r')
    last_ref_block = -1
    for line in f:
        try:
            ref_block = int(line)
        except:
            continue
        total_pg_refs += 1
        if ref_block == last_ref_block:
            continue
        else:
            last_ref_block = ref_block

        pg_fl = alg.pageRequest(ref_block)

        if pg_fl:
            num_pg_fl += 1
    alg.print_stack()
    print(total_pg_refs)
    print(num_pg_fl)
    print(1.0 - num_pg_fl / total_pg_refs)
