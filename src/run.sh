# traces=(2_pools cpp cs cs_long gli multi1 multi2 multi3 ps sprite zigzag)
# traces=(cs_long multi1 multi2 multi3 zigzag)
traces=(cs_long)
start=(5000)
# start=(1000)
# start=(20000)
for i in "${traces[@]}"
do
    for j in "${start[@]}"
    do
        # python3 lirs.py $i
        # python3 ml_lirs_v5_c.py $i $j 1000
        # python3 ml_lirs_v6_1_c.py $i $j 1000
        python3 ml_lirs_v9.py $i $j 1000
        # python3 miss_ratio_graph.py $i _st{$j}_mb{1000}_add_hit
    done
    # python3 miss_ratio_graph_compare.py $i _opt
    # python3 breakdown.py $i
done

# traces=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100)
# trace=(sprite)
# for j in "${trace[@]}"
# do
#         for i in "${traces[@]}"
#         do
#                 python3 ml_lirs_v3.py $j $i 
#         done
# done

# traces=(w103 w100 w098 w097 w095 w083)
# # traces=(w103)
# st=(30000 50000 70000 100000)
# # st=(100000)
# for i in "${traces[@]}"
# do
#         for j in "${st[@]}"
#         do
#                 python3 miss_ratio_graph.py $i $j
#         done
# done
