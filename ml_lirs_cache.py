from collections import deque
from collections import OrderedDict
import codecs
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

def find_lru(s: OrderedDict, pg_tbl: deque):
    temp_s = list(s)
    while pg_tbl[temp_s[0]][2]:
        pg_tbl[temp_s[0]][3] = False
        s.popitem(last=False)
        del temp_s[0]

def get_range(trace) -> int:
    max = 0
    for x in trace:
        if x > max:
            max = x
    return max

def train_model(features, labels, size, rate):
    # Model Training
    temp_features = torch.from_numpy(np.array(features)).type(torch.FloatTensor)
    temp_labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor)

    train_ds = TensorDataset(temp_features, temp_labels)

    batch_size = size
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    next(iter(train_dl))

    model = LogisticRegression(1, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)

    for epoch in range(100):
        for i in enumerate(train_dl):
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(temp_features)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, temp_labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()
    print(loss)
    return model

def LIRS(pg_hits, pg_faults, free_mem, lir_size, lir_stack, hir_stack, pg_table):
    last_ref_block = -1
    is_data = False
    trained = False

    for i in range(len(trace)):
        if is_data:
            m = train_model(training_data, label_data, 30, .01)
            training_data.clear()
            label_data.clear()
            trained = True
            is_data = False

        ref_block = trace[i]

        if ref_block == last_ref_block:
            pg_hits += 1
            continue
        else:
            pass

        if not pg_table[ref_block][1]:
            pg_faults += 1
            if free_mem == 0:
                temp_hir = list(hir_stack)
                pg_table[temp_hir[0]][1] = False
                eviction_list.append(temp_hir[0])
                hir_stack.popitem(last=False)
                free_mem += 1
            elif free_mem > HIR_SIZE:
                pg_table[ref_block][2] = False
                lir_size += 1
            free_mem -= 1
        elif pg_table[ref_block][2]:
            if hir_stack.get(ref_block):
                del hir_stack[ref_block]

        if pg_table[ref_block][1]:
            pg_hits += 1

        #Data gathering. If in LIR stack and non HIR, label = non evictable
        #Else: evictable
        if len(lir_stack) == DATA_COLLECTION_START and not trained:
            for key in lir_stack.keys():
                training_data.append([pg_table[key][4]])
                if pg_table[key][1] and not pg_table[key][2]:
                    label_data.append([1])
                else:
                    label_data.append([0])
            is_data = True

        if lir_stack.get(ref_block):
            counter = 0
            for j in lir_stack.keys():  # Reuse distance
                counter += 1
                if lir_stack[j] == ref_block:
                    break
            pg_table[ref_block][4] = (len(lir_stack) - counter)
            del lir_stack[ref_block]
            find_lru(lir_stack, pg_table)

        pg_table[ref_block][1] = True
        lir_stack[ref_block] = pg_table[ref_block][0]

        if not trained:
            if pg_table[ref_block][2] and pg_table[ref_block][3]:
                pg_table[ref_block][2] = False
                lir_size += 1
                if lir_size > MAX_MEMORY - HIR_SIZE:
                    temp_block = list(lir_stack)[0]
                    pg_table[temp_block][2] = True
                    pg_table[temp_block][3] = False
                    hir_stack[temp_block] = lir_stack[temp_block]
                    find_lru(lir_stack, pg_table)
                    lir_size -= 1
            elif pg_table[ref_block][2]:
                hir_stack[ref_block] = pg_table[ref_block][0]
        else:
            #logic here not working. HIR is not maintained for some reason. Also prediction goes above 1..
            prediction = m(torch.tensor([pg_table[ref_block][4]]).type(torch.FloatTensor))
            if prediction > .5:
                pg_table[ref_block][2] = False
                lir_size += 1
                if lir_size > MAX_MEMORY - HIR_SIZE:
                    temp_block = list(lir_stack)[0]
                    pg_table[temp_block][2] = True
                    pg_table[temp_block][3] = False
                    hir_stack[temp_block] = lir_stack[temp_block]
                    find_lru(lir_stack, pg_table)
                    lir_size -= 1
            else:
                hir_stack[ref_block] = pg_table[ref_block][0]

        pg_table[ref_block][3] = True

        last_ref_block = ref_block


# Read File In
trace = []
with codecs.open("/Users/polskafly/Desktop/REU/LIRS/traces/2_pools.trc", "r", "UTF8") as inputFile:
    inputFile = inputFile.readlines()
for line in inputFile:
    if not line == "*\n":
        trace.append(int(line))

# Init Parameters
MAX_MEMORY = 100
HIR_PERCENTAGE = 1.0
MIN_HIR_MEMORY = 2

HIR_SIZE = MAX_MEMORY * (HIR_PERCENTAGE / 100)
if HIR_SIZE < MIN_HIR_MEMORY:
    HIR_SIZE = MIN_HIR_MEMORY

TRAINING_HIT_PERCENTAGE = 30.0
TRAINING_DATA_MIN = 500
DATA_COLLECTION_START = MAX_MEMORY * 2
if DATA_COLLECTION_START < TRAINING_DATA_MIN:
    DATA_COLLECTION_START = TRAINING_DATA_MIN
# Init End

#Creating stacks and lists
lir_stck = OrderedDict()
hir_stck = OrderedDict()
pg_tbl = deque()
eviction_list = []
training_data = deque()
label_data = deque()

# [Block Number, is_resident, is_hir_block, in_stack, reuse distance]
vm_size = get_range(trace)
for x in range(vm_size + 1):
    pg_tbl.append([x, False, True, False, 9999])

#Creating variables
PG_HITS = 0
PG_FAULTS = 0
free_mem = MAX_MEMORY
lir_size = 0
in_trace = 0
is_trained = False


LIRS(PG_HITS, PG_FAULTS, free_mem, lir_size, lir_stck, hir_stck, pg_tbl)


print("Hits: ", PG_HITS)
print("Faults: ", PG_FAULTS)
print("Total: ", PG_FAULTS + PG_HITS)
print("HIR Size: ", HIR_SIZE)
print("Hit Ratio: ", PG_HITS/(PG_HITS + PG_FAULTS) * 100)

f = open("evictions.txt", "w")
for i in range(len(eviction_list)):
    f.write(str(eviction_list[i]) + "\n")
f.close()

