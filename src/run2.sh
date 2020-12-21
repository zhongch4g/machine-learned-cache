# python3 ml_lirs_v9_pre_train.py zigzag 1000 1000 400
# python3 ml_lirs_v9_pre_train.py zigzag 1000 1000 700
# python3 ml_lirs_v9_pre_train.py zigzag 1000 1000 1000
# python3 ml_lirs_v9_pre_train.py zigzag 1000 1000 1300
# python3 ml_lirs_v9_pre_train.py zigzag 1000 1000 1800
# python3 breakdown.py zigzag 400 700 1000 1300 1800


# python3 ml_lirs_v9_pre_train.py cs_long 5000 1000 400
# python3 ml_lirs_v9_pre_train.py cs_long 5000 1000 1000
# python3 ml_lirs_v9_pre_train.py cs_long 5000 1000 1400
# python3 breakdown.py cs_long 400 1000 1400

traces=(2_pools cpp cs cs_long gli multi1 multi2 multi3 ps sprite zigzag)
# traces=(zigzag_2_pools zigzag_multi1 zigzag_multi2 zigzag_multi3 zigzag_sprite)
# traces=(zigzag_multi1)
# start=(1000 3000 5000 7000)
# start=(3000)
for i in "${traces[@]}"
do
        python3 ml_lirs_v9_pre_train.py $i 1000 1000
        python3 breakdown.py $i
done