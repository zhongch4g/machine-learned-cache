traces=(2_pools cpp cs cs_long gli multi1 multi2 multi3 ps sprite zigzag)
# traces=(zigzag_2_pools zigzag_multi1 zigzag_multi2 zigzag_multi3 zigzag_sprite)
# traces=(zigzag_multi1)
start=(1000 3000 5000 7000)
# start=(3000)
for i in "${traces[@]}"
do
    for j in "${start[@]}"
    do
        # python3 lirs.py $i
        # python3 ml_lirs_v5_c.py $i $j 1000
        # python3 ml_lirs_v6_1.py $i $j 1000
        # python3 ml_lirs_v8.py $i $j 1000
        python3 ml_lirs_v9.py $i $j 1000
        # python3 miss_ratio_graph.py $i _st{$j}_mb{1000}_add_hit
        python3 miss_ratio_graph_compare.py $i _w100_collect_rmax0
    done
done