# traces=(2_pools cpp cs gli multi1 multi2 multi3 ps sprite zigzag)
traces=(sprite)
# start=(1000 3000 5000 7000)
start=(1000 3000)
mini_batch=(1000)
for i in "${traces[@]}"
do
    for j in "${start[@]}"
    do
        for k in "${mini_batch[@]}"
        do
                # python3 lirs.py $i
                python3 ml_lirs_v4.py $i $j $k
                python3 miss_ratio_graph.py $i _st{$j}_mb{$k}_test3
        done
    done
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