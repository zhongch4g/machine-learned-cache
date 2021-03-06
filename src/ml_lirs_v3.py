import sys
import os
import numpy as np
from sklearn.linear_model import SGDClassifier

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
        self.refer_times = 0
        self.reuse_distance = 0
        self.position = 2

class Trace:
    def __init__(self, tName):
        self.trace_path = '../trace/' + tName
        self.parameter_path = '../cache_size/' + tName
        self.trace = []
        self.memory_size = []
        self.vm_size = -1
        self.trace_dict = dict()

    def get_trace(self):
        # print ("get trace from ", self.trace_path)
        with open(self.trace_path, "r") as f:
            for line in f.readlines():
                if not line == "*\n" and not line.strip() == "":
                    block_number = int(line)
                    self.trace.append(block_number)
                    self.trace_dict[block_number] = Node(block_number)
        self.vm_size = max(self.trace)
        return self.trace, self.trace_dict, len(self.trace)

    def get_parameter(self):
        # print ("get trace from ", self.parameter_path)
        with open(self.parameter_path, "r") as f:
            for line in f.readlines():
                if not line == "*\n":
                    self.memory_size.append(int(line.strip()))
        return self.memory_size

class WriteToFile:
    def __init__(self, tName):
        self.tName = tName
        self.FILE = open("../result_set/" + self.tName + "/ml_lirs_" + self.tName, "w")
    
    def write_to_file(self, *args):
        data = ",".join(args)
        # Store the hit&miss ratio
        self.FILE.write(data + "\n")


class LIRS_Replace_Algorithm:
    def __init__(self, vm_size, trace_dict, mem_size, trace_size, model):
        # re-initialize trace_dict
        for k, v in trace_dict.items():
            trace_dict[k] = Node(k)

        self.model = model
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

        self.count_exampler = 0
        self.train = False

        self.predict_times = 0
        self.predict_as_1 = 0
        self.out_stack_hit = 0

        self.mini_batch_X = np.array([])
        self.mini_batch_X = np.array([])

        self.position_importance = {0:1, 1:2, 2:3}

        self.start_use_model = 5000

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

    def get_reuse_distance(self, ref_block):
        ptr = self.Stack_S_Head
        count = 0
        while (ptr):
            count += 1
            if (ptr.block_number == ref_block):
                return count
            ptr = ptr.LIRS_next
        raise("get reuse distance error!")

    
    def print_information(self):
        print ("======== Results ========")
        print ("memory size : ", self.MEMORY_SIZE)
        print ("trace_size : ", self.trace_size)
        print ("Q size : ", self.HIR_SIZE)
        print ("page hit : ", self.page_hit)
        print ("page fault : ", self.page_fault)
        print ("Hit ratio: ", self.page_hit/(self.page_fault + self.page_hit) * 100)
        print ("Out stack hit : ", self.out_stack_hit)
        if (self.predict_times > 0):
            print ("Predict Times =", self.predict_times, "Predict 1 ratio =", self.predict_as_1/self.predict_times * 100)
        return self.MEMORY_SIZE, self.page_fault/(self.page_fault + self.page_hit) * 100, self.page_hit/(self.page_fault + self.page_hit) * 100
    
    def print_stack(self, v_time):
        if (v_time < 50000):
            return 
        ptr = self.Stack_S_Head
        while (ptr):
            print("R" if ptr == self.Rmax0 else "", end="")
            # print("H" if ptr.is_hir else "L", end="")
            # print("R" if ptr.is_resident else "N", end="")
            print(ptr.block_number, end="-")
            ptr = ptr.LIRS_next
        print ()

    def LIRS(self, v_time, ref_block):
        # print (v_time, ref_block, self.page_table[ref_block].recency, self.page_table[ref_block].reuse_distance)
        if not self.page_table[ref_block].recency:
            self.out_stack_hit += 1
            # print("out stack hit", v_time, ref_block)

        if (ref_block == self.last_ref_block):
            self.page_hit += 1
            return
        self.last_ref_block = ref_block
        self.page_table[ref_block].refer_times += 1
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

        if (self.page_table[ref_block].refer_times > 1):
            self.count_exampler += 1
            p_feature = self.position_importance[self.page_table[ref_block].position]
            
            # print (v_time, "train = ", np.array([[self.page_table[ref_block].reuse_distance, p_feature] + [1 if i == self.page_table[ref_block].position else 0 for i in range(3)]]), np.array([1 if self.page_table[ref_block].recency else -1]))
            # self.model.partial_fit(np.array([[self.page_table[ref_block].reuse_distance, p_feature] + [1 if i == self.page_table[ref_block].position else 0 for i in range(3)]]), np.array([1 if self.page_table[ref_block].recency else -1], ), classes = np.array([1, -1]))
            if (self.mini_batch_X.all()):
                self.mini_batch_X = np.array([[1 if i == self.page_table[ref_block].position else 0 for i in range(3)]])
                self.mini_batch_Y = np.array([1 if self.page_table[ref_block].recency else -1])
            else:
                # print(self.mini_batch_X)
                # self.mini_batch_X = np.r_[self.mini_batch_X, [[p_feature] + [1 if i == self.page_table[ref_block].position else 0 for i in range(3)]]]
                self.mini_batch_X = np.r_[self.mini_batch_X, [[1 if i == self.page_table[ref_block].position else 0 for i in range(3)]]]
                self.mini_batch_Y = np.r_[self.mini_batch_Y, [1 if self.page_table[ref_block].recency else -1]]
            if (len(self.mini_batch_X) == 1000):
                self.model.partial_fit(self.mini_batch_X, self.mini_batch_Y, classes = np.array([1, -1]))
                self.mini_batch_X = np.array([])
                self.mini_batch_Y = np.array([])
            
            # type feature
            p_type = -1
            if (self.page_table[ref_block].recency0):
                p_type = 0
            elif (self.page_table[ref_block].recency):
                p_type = 1
            else:
                p_type = 2
            self.page_table[ref_block].position = p_type
            # self.page_table[ref_block].reuse_distance = self.get_reuse_distance(ref_block)  # Setting the reuse distance for that block
        
        # when use the data to predict the model
        if (self.count_exampler == self.start_use_model):
            self.train = True

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
            if (not self.train):
                self.add_stack_Q(ref_block) # func():
            else:
                p_feature = self.position_importance[self.page_table[ref_block].position]

                # feature = np.array([[self.page_table[ref_block].reuse_distance, p_feature] + [1 if i == self.page_table[ref_block].position else 0 for i in range(3)], ])
                feature = np.array([[p_feature] + [1 if i == self.page_table[ref_block].position else 0 for i in range(3)], ])
                feature = np.array([[1 if i == self.page_table[ref_block].position else 0 for i in range(3)], ])
                # print ([self.page_table[ref_block].reuse_distance, p_feature] + [1 if i == self.page_table[ref_block].position else 0 for i in range(3)])
                prediction = self.model.predict(feature.reshape(1, -1))
                self.predict_times += 1
                if prediction == 1:
                    self.predict_as_1 += 1
                    self.page_table[ref_block].is_hir = False
                    self.add_stack_Q(self.Rmax.block_number) # func():
                    self.Rmax.is_hir = True
                    self.Rmax.recency = False
                    self.find_new_Rmax()
                else:
                    # print ("Predict HIR")
                    self.add_stack_Q(ref_block) # func():

        self.page_table[ref_block].recency = True
        self.page_table[ref_block].recency0 = True

        # self.print_stack(v_time)



def main(tName): 
    # result file
    FILE = WriteToFile(tName)
    # get trace
    trace_obj = Trace(tName)
    # get the trace
    trace, trace_dict, trace_size = trace_obj.get_trace()
    memory_size = trace_obj.get_parameter()
    # memory_size = [1000]
    for memory in memory_size:
        model = SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)
        lirs_replace = LIRS_Replace_Algorithm(trace_obj.vm_size, trace_dict, memory, trace_size, model)
        for v_time, ref_block in enumerate(trace):
            lirs_replace.LIRS(v_time, ref_block)

        memory_size, miss_ratio, hit_ratio = lirs_replace.print_information()
        FILE.write_to_file(str(memory_size), str(miss_ratio), str(hit_ratio))
    

if __name__=="__main__": 
    if (len(sys.argv) != 2):
        raise("usage: python3 XXX.py trace_name")
    tName = sys.argv[1]
    main(tName)



node = Node(5)

node1 = node

node.next = 1
node.prev = 2
