# traces=(2_pools cpp cs gli multi1 multi2 multi3 ps sprite zigzag)
# # traces=(gli)
# for i in "${traces[@]}"
# do
#         # python3 lirs.py $i
#         # python3 ml_lirs.py $i
#         python3 in_stack_miss_graph.py $i
#         python3 miss_ratio_graph.py $i
# done

traces=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100)
trace=(sprite)
for j in "${trace[@]}"
do
        for i in "${traces[@]}"
        do
                python3 ml_lirs_v3.py $j $i 
        done
done