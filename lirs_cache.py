from collections import deque
from collections import OrderedDict
import codecs

#Two Stacks: S and Q
def find_LIR(S: OrderedDict, pg: deque):
    while list(S.items())[0][1][2]:
        pg[list(S.items())[0][0]][3] = False
        S.popitem(last=False)
    return S

def set_hir_block(pg_table, ref_block, bool):
    pg_table[ref_block][2] = bool

def set_residence(pg_table, ref_block, bool):
    pg_table[ref_block][1] = bool

def set_recency(pg_table, ref_block, bool):
    pg_table[ref_block][3] = bool

def get_range(trace) -> int:
    min = 0
    max = 0

    for x in trace:
        if x > max:
            max = x
        if x <= min:
            min = x

    return max

#Read File In
trace = []
with codecs.open("/Users/polskafly/Desktop/REU/LIRS/traces/sprite.trc", "r", "UTF8") as inputFile:
    inputFile = inputFile.readlines()
for line in inputFile:
    trace.append(int(line))

#Initialization Parameters
MAX_MEMORY = 1000
HIR_PERCENTAGE = 1.0
MIN_HIR_MEMORY = 2

HIR_SIZE = MAX_MEMORY * (HIR_PERCENTAGE/(100))
if HIR_SIZE < MIN_HIR_MEMORY:
    HIR_SIZE = MIN_HIR_MEMORY
#Init End

lir_stack = OrderedDict()
hir_stack = OrderedDict()
pg_table = deque()

#[Block Number, is_resident, is_hir_block, in_stack]
vm_size = get_range(trace)
for x in range(vm_size+1):
    pg_table.append([x, False, True, False])

PG_HITS = 0
PG_FAULTS = 0
free_mem = MAX_MEMORY
lir_size = 0

#Possible big issue. I don't move HIR stack objects into LIR stack.

for i in range(len(trace)):
    ref_block = trace[i]
    if pg_table[ref_block][1]:
        PG_HITS += 1
        #if not a HIR resident and is in stack S
        if not pg_table[ref_block][2] and pg_table[ref_block][3]:
            lir_stack.move_to_end(ref_block)
        elif pg_table[ref_block][2] and pg_table[ref_block][3]:
            lir_stack.move_to_end(ref_block)
            set_hir_block(pg_table, ref_block, False)
            lir_stack[ref_block] = pg_table[ref_block]
            if ref_block in hir_stack:
                hir_stack.move_to_end(ref_block)
                hir_stack.popitem(last=True)
            #Check if HIR is full before popping if I don't get same results as prof.
            if len(hir_stack) == HIR_SIZE:
                temp = hir_stack.popitem(last=False)
                print(i, " ", temp)
            set_recency(pg_table, list(lir_stack.keys())[0], False)
            set_hir_block(pg_table, list(lir_stack.keys())[0], True)
            hir_stack[list(lir_stack.keys())[0]] = pg_table[list(lir_stack.keys())[0]]
            lir_stack.move_to_end(list(lir_stack.keys())[0])
            lir_stack.popitem(last=True)

        elif pg_table[ref_block][2] and not pg_table[ref_block][3]:
            hir_stack.move_to_end(ref_block)
            hir_stack[ref_block] = pg_table[ref_block]
        find_LIR(lir_stack, pg_table)
    else:
        PG_FAULTS += 1
        lir_stack[ref_block] = pg_table[ref_block]
        lir_size += 1
        if free_mem == 0:
            set_residence(pg_table, list(hir_stack.keys())[0], False)
            temp = hir_stack.popitem(last=False)
            print(i, " ", temp)
            if pg_table[ref_block][3] and pg_table[ref_block][2]:
                set_recency(pg_table, list(lir_stack.keys())[0], False)
                set_hir_block(pg_table, list(lir_stack.keys())[0], True)
                hir_stack[list(lir_stack.keys())[0]] = pg_table[list(lir_stack.keys())[0]]
                lir_stack.move_to_end(list(lir_stack.keys())[0])
                lir_stack.popitem(last=True)

                set_residence(pg_table, ref_block, True)
                set_hir_block(pg_table, ref_block, False)
                lir_stack[ref_block] = pg_table[ref_block]
                lir_size -= 1
            elif not pg_table[ref_block][3] and pg_table[ref_block][2]:
                #Problem
                set_residence(pg_table, ref_block, True)
                set_recency(pg_table, ref_block, True)
                lir_size -= 1
                lir_stack[ref_block] = pg_table[ref_block]
                hir_stack[ref_block] = pg_table[ref_block]
            free_mem += 1
        elif free_mem > HIR_SIZE:
            set_hir_block(pg_table, ref_block, False)
            set_recency(pg_table, ref_block, True)
            set_residence(pg_table, ref_block, True)
            lir_stack[ref_block] = pg_table[ref_block]
        elif lir_size > MAX_MEMORY - HIR_SIZE:
            set_recency(pg_table, ref_block, False)
            set_hir_block(pg_table, ref_block, True)
            set_residence(pg_table, ref_block, True)
            hir_stack[ref_block] = pg_table[ref_block]
            lir_stack[ref_block] = pg_table[ref_block]
            lir_size -= 1


        free_mem -= 1
        find_LIR(lir_stack, pg_table)

print(PG_HITS)
print(PG_FAULTS)
print(PG_FAULTS+PG_HITS)