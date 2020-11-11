import sys
import os


class Node:
    def __init__(self, b_num):
        self.block_number = b_num
        self.is_resident = False
        self.is_hir = True
        self.LIRS_prev = None
        self.LIRS_next = None
        self.HIR_prev = None
        self.HIR_next = None
        self.recency = False # Rmax boundary
        self.recency0 = False # LRU boundary

class Trace:
    def __init__(self, tName):
        self.trace_path = '../trace/' + tName
        self.parameter_path = '../cache_size/' + tName
        self.trace = []
        self.memory_size = []
        self.vm_size = -1
        self.trace_dict = dict()

    def get_trace(self):
        print ("get trace from ", self.trace_path)
        with open(self.trace_path, "r") as f:
            for line in f.readlines():
                if not line == "*\n" and not line.strip() == "":
                    block_number = int(line)
                    self.trace.append(block_number)
                    self.trace_dict[block_number] = Node(block_number)
        self.vm_size = max(self.trace)
        return self.trace, self.trace_dict, len(self.trace)

    def get_parameter(self):
        print ("get trace from ", self.parameter_path)
        with open(self.parameter_path, "r") as f:
            for line in f.readlines():
                if not line == "*\n":
                    self.memory_size.append(int(line))
        return self.memory_size

class LIRS_Replace_Algorithm:
    def __init__(self, vm_size, trace_dict, mem_size, trace_size):
        self.trace_size = trace_size
        self.page_table = trace_dict # {block number : Node}
        self.MEMORY_SIZE = mem_size
        self.Free_Memory_Size = mem_size
        HIR_PERCENTAGE = 1.0
        MIN_HIR_MEMORY = 2
        self.HIR_SIZE = mem_size * (HIR_PERCENTAGE / 100)
        if self.HIR_SIZE < MIN_HIR_MEMORY:
            self.HIR_SIZE = MIN_HIR_MEMORY
        
        self.Stack_S_Head = None
        self.Sack_S_Tail = None
        self.Stack_Q_Head = None
        self.Stack_Q_Tail = None

        self.Rmax = None
        self.Rmax0 = None

        self.page_fault = 0
        self.page_hit = 0

        self.last_ref_block = -1

    def remove_stack_Q(self, b_num):
        if (not self.page_table[b_num].HIR_next and not self.page_table[b_num].HIR_prev):
            self.Stack_Q_Tail, self.Stack_Q_Head = None, None
        elif (self.page_table[b_num].HIR_next and self.page_table[b_num].HIR_prev):
            self.page_table[b_num].HIR_prev.HIR_next = self.page_table[b_num].HIR_next
            self.page_table[b_num].HIR_next.HIR_prev = self.page_table[b_num].HIR_prev
            self.page_table[b_num].HIR_next, self.page_table[b_num].HIR_prev = None, None
        elif (not self.page_table[b_num].HIR_prev):
            self.Stack_Q_Head = self.page_table[b_num].HIR_next
            self.Stack_Q_Head.HIR_prev.HIR_next = None
            self.Stack_Q_Head.HIR_prev = None
        elif (not self.page_table[b_num].HIR_next):
            self.Stack_Q_Tail = self.page_table[b_num].HIR_prev
            self.Stack_Q_Tail.HIR_next.HIR_prev = None
            self.Stack_Q_Tail.HIR_next = None
        else:
            raise("Stack Q remove error \n")
        return True

    def add_stack_Q(self, b_num):
        if (not self.Stack_Q_Head):
            self.Stack_Q_Head = self.Stack_Q_Tail = self.page_table[b_num]
        else:
            self.page_table[b_num].HIR_next = self.Stack_Q_Head
            self.Stack_Q_Head.HIR_prev = self.page_table[b_num]
            self.Stack_Q_Head = self.page_table[b_num]

    def remove_stack_S(self, b_num):
        if (not self.page_table[b_num].LIRS_prev and not self.page_table[b_num].LIRS_next):
            return False

        if (self.page_table[b_num] == self.Rmax):
            self.Rmax = self.page_table[b_num].LIRS_prev
            self.find_new_Rmax()

        if (self.page_table[b_num] == self.Rmax0):
            self.Rmax0 = self.page_table[b_num].LIRS_prev

        if (self.page_table[b_num].LIRS_prev and self.page_table[b_num].LIRS_next):
            self.page_table[b_num].LIRS_prev.LIRS_next = self.page_table[b_num].LIRS_next
            self.page_table[b_num].LIRS_next.LIRS_prev = self.page_table[b_num].LIRS_prev
            self.page_table[b_num].LIRS_prev, self.page_table[b_num].LIRS_next = None, None
        elif (not self.page_table[b_num].LIRS_prev):
            self.Stack_S_Head = self.page_table[b_num].LIRS_next
            self.Stack_S_Head.LIRS_prev.LIRS_next = None
            self.Stack_S_Head.LIRS_prev = None
        elif (not self.page_table[b_num].LIRS_next):
            self.Stack_S_Tail = self.page_table[b_num].LIRS_prev
            self.Stack_S_Tail.LIRS_next.LIRS_prev = None
            self.Stack_S_Tail.LIRS_next = None
        else:
            raise("Stack S remove error \n")
        return True

    def add_stack_S(self, b_num):
        if (not self.Stack_S_Head):
            self.Stack_S_Head = self.Stack_S_Tail = self.page_table[b_num]
            self.Rmax = self.page_table[b_num]
        else:
            self.page_table[b_num].LIRS_next = self.Stack_S_Head
            self.Stack_S_Head.LIRS_prev = self.page_table[b_num]
            self.Stack_S_Head = self.page_table[b_num]
        return True

    def find_new_Rmax(self):
        if (not self.Rmax):
            raise("Warning Rmax0 \n")
        while (self.Rmax.is_hir == True):
            self.Rmax.recency = False
            self.Rmax = self.Rmax.LIRS_prev

    def find_new_Rmax0(self):
        if (not self.Rmax0):
            raise("Warning Rmax0 \n")
        self.Rmax0.recency0 = False
        self.Rmax0 = self.Rmax0.LIRS_prev
        return self.Rmax0
    
    def print_information(self):
        print ("memory size : ", self.MEMORY_SIZE)
        print ("trace_size : ", self.trace_size)
        print ("Q size : ", self.HIR_SIZE)
        print ("page hit : ", self.page_hit)
        print ("page fault : ", self.page_fault)
        print ("Hit ratio: ", self.page_hit/(self.page_fault + self.page_hit))
    
    def print_stack(self, vtime):
        if (5000 > vtime or vtime > 5003):
            return 
        ptr = self.Stack_S_Head
        while (ptr):
            # print("R" if ptr == self.Rmax else "", end="")
            print("H" if ptr.is_hir else "L", end="")
            print("R" if ptr.is_resident else "N", end="")
            print(ptr.block_number, end="-")
            ptr = ptr.LIRS_next
        print ()


    def LIRS(self, v_time, ref_block):
        if (ref_block == self.last_ref_block):
            self.page_hit += 1
            return
        self.last_ref_block = ref_block
        if(not self.page_table[ref_block].is_resident):
            self.page_fault += 1
            # print (v_time, ref_block, 0)
            # self.print_stack(v_time)

            if (self.Free_Memory_Size == 0):
                self.Stack_Q_Tail.is_resident = False
                # self.page_table[self.Stack_Q_Tail.block_number].is_resident = False
                self.remove_stack_Q(self.Stack_Q_Tail.block_number) # func(): remove block in the tail of stack Q
                self.Free_Memory_Size += 1
            elif (self.Free_Memory_Size > self.HIR_SIZE):
                self.page_table[ref_block].is_hir = False
            
            self.Free_Memory_Size -= 1
        elif (self.page_table[ref_block].is_hir):
            self.remove_stack_Q(ref_block) # func(): 

        if(self.page_table[ref_block].is_resident):
            self.page_hit += 1
            # print (v_time, ref_block, 1, "HIR" if self.page_table[ref_block].is_hir else "LIR")
            # self.print_stack(v_time)
        
        # find new Rmax0
        if (self.Rmax0 and not self.page_table[ref_block].recency0):
            if (self.Rmax0 != self.page_table[ref_block]):
                self.find_new_Rmax0()
        
        self.remove_stack_S(ref_block)
        self.add_stack_S(ref_block)

        if (self.Free_Memory_Size == 0 and not self.Rmax0):
            self.Rmax0 = self.page_table[self.Rmax.block_number]
        
        self.page_table[ref_block].is_resident = True

        if (self.page_table[ref_block].is_hir and self.page_table[ref_block].recency):
            self.page_table[ref_block].is_hir = False

            self.add_stack_Q(self.Rmax.block_number) # func():
            self.Rmax.is_hir = True
            self.Rmax.recency = False
            self.find_new_Rmax()
        elif (self.page_table[ref_block].is_hir):
            self.add_stack_Q(ref_block) # func():

        self.page_table[ref_block].recency = True
        self.page_table[ref_block].recency0 = True

        # self.print_stack()



def main(tName): 
    # get trace
    trace_obj = Trace(tName)
    # get the trace
    trace, trace_dict, trace_size = trace_obj.get_trace()
    memory_size = trace_obj.get_parameter()
    memory_size = [100]
    for memory in memory_size:
        lirs_replace = LIRS_Replace_Algorithm(trace_obj.vm_size, trace_dict, memory, trace_size)
        for v_time, ref_block in enumerate(trace):
            lirs_replace.LIRS(v_time, ref_block)
        lirs_replace.print_information()
    


    

if __name__=="__main__": 
    if (len(sys.argv) != 2):
        raise("usage: python3 XXX.py trace_name")
    tName = sys.argv[1]

    main(tName)



node = Node(5)

node1 = node

node.next = 1
node.prev = 2
