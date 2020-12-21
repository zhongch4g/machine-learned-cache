"""
model: naive bayes sklearn.naive_bayes.BernoulliNB

training data collection
1. look prev 100 blocks

time to predict: use start time parameter

feature:

"""

import sys
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

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
        self.is_last_hit = False
        """
        in lru, in lir stack, out lir stack 
        """
        self.position = [0]
        self.last_is_hir = True

class Trace:
    def __init__(self, t_name, type_of_trace):
        if (type_of_trace == "lirs_trace"):
            self.trace_path = '../trace/' + t_name
            self.parameter_path = '../cache_size/' + t_name
        else:
            self.trace_path = '/home/zhongchen/myLirs/result_set/' + t_name + '/LIRS2_SEQ-DLIRS_TRACE'
            self.parameter_path = '/home/zhongchen/myLirs/trace_parameter/' + t_name

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
    def __init__(self, tName, st):
        self.tName = tName
        try:  
            os.mkdir("../result_set/" + self.tName + "/")  
        except OSError as error:  
            print(error)
        self.FILE = open("../result_set/" + self.tName + "/ml_lirs_v9" + str(st) + "_" + self.tName, "w")

    def write_to_file(self, *args):
        data = ",".join(args)
        # Store the hit&miss ratio
        self.FILE.write(data + "\n")

    

class Segment_Miss:
    def __init__(self, tName, cache, args):
        self.tName = tName
        self.FILE = open("../result_set/" + self.tName + "/ml_lirs_v9_" + self.tName + "_" + str(cache) + "_segment_miss_" + args, "w")
    def record(self, segment_miss_ratio):
        self.FILE.write(str(segment_miss_ratio) + "\n")

class Breakdown:
    def __init__(self, tName, cache, args):
        self.tName = tName
        self.FILE = open("../result_set/" + self.tName + "/ml_lirs_v9_" + self.tName + "_" + str(cache) + "_breakdown_" + args, "w")

    def record(self, *args):
        # print(list(args))
        self.FILE.write(",".join(list(args)) + "\n")
        self.FILE.flush()


class LIRS_Replace_Algorithm:
    def __init__(self, pretrain, training_data, args, t_name, trace, vm_size, trace_dict, mem_size, trace_size, model, start_use_model=3000, mini_batch=1000):
        self.trace_name = t_name
        self.Cache_Size = mem_size
        self.trace = trace
        self.pre_train = pretrain
        self.training_data = training_data
        self.model = model
        if (self.training_data):
            self.training_data = np.array(training_data)
            # print(self.training_data[:, 0].reshape(-1, 1), len(self.training_data[:, 0]))
            # print(self.training_data[:, 1], len(self.training_data[:, 1]))
            self.model.partial_fit(self.training_data[:, 0].reshape(-1, 1), self.training_data[:, 1], classes = np.array([1, 0]))

        # re-initialize trace_dict
        for k, v in trace_dict.items():
            trace_dict[k] = Node(k)

        self.trace_size = trace_size
        self.page_table = trace_dict # {block number : Node}
        self.MEMORY_SIZE = mem_size
        self.Free_Memory_Size = mem_size
        self.Window_Size = 100
        HIR_PERCENTAGE = 1.0
        MIN_HIR_MEMORY = 2
        self.HIR_SIZE = mem_size * (HIR_PERCENTAGE / 100)
        if self.HIR_SIZE < MIN_HIR_MEMORY:
            self.HIR_SIZE = MIN_HIR_MEMORY
        
        self.Stack_S_Head = None
        self.Sack_S_Tail = None
        self.Stack_Q_Head = None
        self.Stack_Q_Tail = None

        self.lir_size = 0
        self.Rmax = None
        self.Rmax0 = None

        self.page_fault = 0
        self.page_hit = 0

        self.last_ref_block = -1

        self.count_exampler = 0
        self.train = False

        self.predict_times = 0
        self.predict_H_L = 0
        self.predict_L_H = 0
        self.predict_H_H = 0
        self.predict_L_L = 0
        self.out_stack_hit = 0

        self.mini_batch_X = np.array([])
        self.mini_batch_X = np.array([])

        self.start_use_model = start_use_model
        self.mini_batch = mini_batch

        # segment miss
        self.epoch = self.trace_size // 20
        print(self.epoch)
        self.last_miss = 0
        self.seg_file = Segment_Miss(t_name, mem_size, args)
        self.breakdown = Breakdown(t_name, mem_size, args)
        self.positive_sampler = 0
        self.negative_sampler = 0

        self.in_in = 0
        self.out_in = 0
        self.in_out = 0
        self.out_out = 0

        self.hit = False

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

            # record is hir status
            self.Rmax.last_is_hir = True

            self.Rmax = self.Rmax.LIRS_prev

    def find_new_Rmax0(self):
        if (not self.Rmax0):
            raise("Warning Rmax0 \n")
        self.Rmax0.recency0 = False
        # if (not self.Rmax0.is_hir):
        #     self.collect_sampler(self.Rmax0.block_number, -1, [1])
        # else:
        #     self.collect_sampler(self.Rmax0.block_number, -1, [0])

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

    def resident_number(self):
        count = 0
        for b_num, node in self.page_table.items():
            if (node.is_resident):
                count += 1
        print (self.MEMORY_SIZE, count)

    def collect_sampler(self, v_time, ref_block, feature):
        self.count_exampler += 1

        """
        record training data
        """
        label = 0
        # Search LRU size from feature
        for i in range(1, 1 + self.Cache_Size):
            # print(ref_block, self.trace[v_time + i])
            if (v_time + i < self.trace_size and ref_block != self.trace[v_time + i]):
                continue
            label = 1
            break
        # print("break")

        if (feature == 1):
            if (label == 1):
                self.in_in += 1
            else:
                self.in_out += 1
        else:
            if (label == 1):
                self.out_in += 1
            else:
                self.out_out += 1

        self.training_data.append([feature, label])
        # print (feature, label)
    

    def print_information(self):
        print ("======== Results ========")
        print ("trace : ", self.trace_name)
        print ("memory size : ", self.MEMORY_SIZE)
        print ("trace_size : ", self.trace_size)
        print ("Q size : ", self.HIR_SIZE)
        print ("page hit : ", self.page_hit)
        print ("page fault : ", self.page_fault)
        print ("Hit ratio: ", self.page_hit/(self.page_fault + self.page_hit) * 100)
        print ("Out stack hit : ", self.out_stack_hit)
        # print ("Positive sampler: ", self.positive_sampler, "Negative sampler: ", self.negative_sampler)
        print ("Predict Times =", self.predict_times, "H->L", self.predict_H_L, "L->H", self.predict_L_H, "H->H", self.predict_H_H, "L->L", self.predict_L_L)
        return self.MEMORY_SIZE, self.page_fault/(self.page_fault + self.page_hit) * 100, self.page_hit/(self.page_fault + self.page_hit) * 100, self.training_data
    
    def print_stack(self, v_time):
        ptr = self.Stack_S_Head
        while (ptr):
            print("R" if ptr == self.Rmax0 else "", end="")
            # print("H" if ptr.is_hir else "L", end="")
            # print("r" if ptr.is_resident else "N", end="")
            print("r" if ptr.recency0 else "&", end="")
            print(ptr.block_number, end="-")
            ptr = ptr.LIRS_next
        print ()

    def check_Rmax0(self):
        # ptr = self.Stack_S_Head
        # count = 1
        # while (ptr != self.Rmax0):
        #     count += 1
        #     ptr = ptr.LIRS_next
        # print (count)
        
        ptr = self.Stack_Q_Head
        count = 1
        while (ptr):
            count += 1
            ptr = ptr.HIR_next
        print (count)

    def LIRS(self, v_time, ref_block):
        self.hit = False
        if (v_time > 0 and v_time % self.epoch == 0):
            segment_miss = self.page_fault - self.last_miss
            self.seg_file.record(segment_miss/self.epoch * 100)
            # print(segment_miss/self.epoch * 100)
            self.last_miss = self.page_fault
            # print(self.lir_size)
            # self.check_Rmax0()
            # print (str(self.out_out), str(self.out_in), str(self.in_out), str(self.in_in))
            self.breakdown.record(str(self.out_out + self.out_in + self.in_out + self.in_in), str(self.out_out), str(self.out_in), str(self.in_out), str(self.in_in))
            self.out_out = 0
            self.out_in = 0
            self.in_out = 0
            self.in_in = 0

        if not self.page_table[ref_block].recency:
            self.out_stack_hit += 1
  
        if (ref_block == self.last_ref_block):
            self.page_hit += 1
            print("Return : ", ref_block, self.last_ref_block)
            return

        self.last_ref_block = ref_block
        self.page_table[ref_block].refer_times += 1
        if(not self.page_table[ref_block].is_resident):
            self.page_fault += 1
            if (self.Free_Memory_Size == 0):
                self.Stack_Q_Tail.is_resident = False
                self.remove_stack_Q(self.Stack_Q_Tail.block_number) # func(): remove block in the tail of stack Q
                self.Free_Memory_Size += 1
                self.Window_Size += 1
            elif (self.Free_Memory_Size > self.HIR_SIZE):
                self.page_table[ref_block].is_hir = False
                self.lir_size += 1
            self.Window_Size -= 1
            self.Free_Memory_Size -= 1
        elif (self.page_table[ref_block].is_hir):
            self.remove_stack_Q(ref_block) # func(): 

        if(self.page_table[ref_block].is_resident):
            self.page_hit += 1
            self.hit = True
        
        # find new Rmax0
        if (self.Rmax0 and not self.page_table[ref_block].recency0):
            if (self.Rmax0 != self.page_table[ref_block]):
                self.find_new_Rmax0()

        if (not self.pre_train):
            if (self.page_table[ref_block].recency0):
                # collect positive sampler
                self.collect_sampler(v_time, ref_block, 1)
            else:
                self.collect_sampler(v_time, ref_block, 0)

        self.remove_stack_S(ref_block)
        self.add_stack_S(ref_block)

        if (self.Free_Memory_Size == 0 and not self.Rmax0):
            self.Rmax0 = self.page_table[self.Rmax.block_number]
        
        self.page_table[ref_block].is_resident = True

        
        if (self.pre_train):
            feature = np.array([self.page_table[ref_block].recency0])
            prediction = self.model.predict(feature.reshape(1, -1))
            # print(prediction, prediction[0])
            if (self.page_table[ref_block].recency0):
                if (prediction[0] == 1):
                    self.in_in += 1
                else:
                    self.in_out += 1
            else:
                if (prediction[0] == 1):
                    self.out_in += 1
                else:
                    self.out_out += 1
        

        # start predict
        if (self.train):
            # feature = np.array([self.page_table[ref_block].position])
            feature = np.array([self.page_table[ref_block].recency0])
            prediction = self.model.predict(feature.reshape(1, -1))
            self.predict_times += 1
            if (self.page_table[ref_block].is_hir):
                if (prediction == 1 and self.Cache_Size-self.lir_size > self.HIR_SIZE): 
                    # H -> L
                    self.lir_size += 1
                    self.predict_H_L += 1
                    self.page_table[ref_block].is_hir = False
                    self.add_stack_Q(self.Rmax.block_number) # func():
                    self.Rmax.is_hir = True
                    self.lir_size -= 1
                    self.Rmax.recency = False
                    self.find_new_Rmax()
                elif (prediction == -1):
                    # H -> H
                    self.predict_H_H += 1
                    self.add_stack_Q(ref_block) # func():
            else:
                # print(prediction, self.lir_size, self.HIR_SIZE)
                # if prev LIR
                if (prediction == -1 and self.lir_size > self.HIR_SIZE):
                    # L -> H
                    self.predict_L_H += 1
                    self.lir_size -= 1
                    # print (v_time, ref_block, self.lir_size, "L -> H")
                    self.add_stack_Q(ref_block) # func():
                    if (ref_block == self.Rmax.block_number):
                        self.Rmax.is_hir = True
                        self.Rmax.recency = False
                        self.find_new_Rmax()
                    else:
                        self.page_table[ref_block].is_hir = True

                elif (prediction == 1):
                    # L -> L do nothing
                    self.predict_L_L += 1
                    pass
        else:
            # origin lirs
            if (self.page_table[ref_block].is_hir and self.page_table[ref_block].recency):
                self.page_table[ref_block].is_hir = False
                self.lir_size += 1
                
                self.add_stack_Q(self.Rmax.block_number) # func():
                self.Rmax.is_hir = True
                self.lir_size -= 1
                self.Rmax.recency = False

                self.find_new_Rmax()
            elif (self.page_table[ref_block].is_hir):
                self.add_stack_Q(ref_block) # func():

        self.page_table[ref_block].recency = True
        self.page_table[ref_block].recency0 = True
        self.page_table[ref_block].refer_times += 1
        self.page_table[ref_block].last_is_hir = self.page_table[ref_block].is_hir
        self.page_table[ref_block].is_last_hit = self.hit
        # if (v_time % 100 == 1):
            # print ("lir_size : ", self.lir_size)
            # self.resident_number()
        # print("lir size: ", self.lir_size)
        # if (v_time < 11):
        #     self.print_stack(v_time)

def pre_train(model, path):
    with open("./" + path, "r") as f:
        for line in f.readlines():
            line = line.split(" ")
            feature = int(line[0])
            label = int(line[1])
            model.partial_fit(np.array([[feature]]), np.array([label]), classes = np.array([1, 0]))
    return model

def main(t_name, start_predict, mini_batch, cache=None): 
    # result file
    FILE = WriteToFile(t_name, start_predict)
    # get trace(lirs_trace/lirs2_trace)
    trace_obj = Trace(t_name, "lirs_trace")
    # get the trace
    trace, trace_dict, trace_size = trace_obj.get_trace()
    memory_size = None
    if (not cache):
        memory_size = trace_obj.get_parameter()
    else:
        memory_size = [cache]
    # print(memory_size)
    for memory in memory_size:
        model = BernoulliNB()
        # opt
        lirs_replace = LIRS_Replace_Algorithm(False, [], "opt", t_name, trace, trace_obj.vm_size, trace_dict, memory, trace_size, model, start_predict, mini_batch)
        for v_time, ref_block in enumerate(trace):
            lirs_replace.LIRS(v_time, ref_block)

        memory_size, miss_ratio, hit_ratio, training_data = lirs_replace.print_information()
        print("Finish OPT")
        # ml lirs
        model = BernoulliNB()
        lirs_replace = LIRS_Replace_Algorithm(True, training_data, "ml_lirs", t_name, trace, trace_obj.vm_size, trace_dict, memory, trace_size, model, start_predict, mini_batch)
        for v_time, ref_block in enumerate(trace):
            lirs_replace.LIRS(v_time, ref_block)

        memory_size, miss_ratio, hit_ratio, training_data = lirs_replace.print_information()

        FILE.write_to_file(str(memory_size), str(miss_ratio), str(hit_ratio))


    

if __name__=="__main__": 
    # if (len(sys.argv) != 4):
    #     raise("usage: python3 XXX.py trace_name start_predict mini_batch")
    t_name = sys.argv[1]
    start_predict = int(sys.argv[2])
    mini_batch = int(sys.argv[3])
    cache = None
    if (len(sys.argv) == 5):
        cache = int(sys.argv[4])
    main(t_name, start_predict, mini_batch, cache)

