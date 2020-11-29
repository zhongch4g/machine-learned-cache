# traces=(multi3 ps sprite zigzag)
# traces=(zigzag)
# start=(1000 3000 5000 7000)
# start=(1000)
# for i in "${traces[@]}"
# do
#     for j in "${start[@]}"
#     do
#         # python3 lirs.py $i
#         python3 ml_lirs_v5.py $i $j 1000
#         python3 miss_ratio_graph.py $i _st{$j}_mb{1000}_is_hir_feature
#     done
# done

# traces=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100)
# trace=(sprite)
# for j in "${trace[@]}"
# do
#         for i in "${traces[@]}"
#         do
#                 python3 ml_lirs_v3.py $j $i 
#         done
# done

# traces=(sprite)
# for i in "${traces[@]}"
# do
#     python3 ml_lirs_v5.py $i 1000 1000
#     python3 lirs.py $i
#     python3 segment_miss_graph.py $i _800_start1000
# done
traces=(w103 w079)
for i in "${traces[@]}"
do
    python3 ml_lirs_v5.py $i 1000 1000
    # python3 miss_ratio_graph.py $i _st{1000}_mb{1000}_is_hir_feature
done