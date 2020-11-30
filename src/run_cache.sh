# traces=(2_pools cpp cs cs_long gli multi1 multi2)
cache=(300)
trace=cs_long
for j in "${cache[@]}"
do
    python3 lirs.py $trace $j
    python3 ml_lirs_v6.py $trace 3000 1000 $j
    python3 segment_miss_graph.py $trace _c{$j}_st{3000}
done
