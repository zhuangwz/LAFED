GPUID=6

cd ..

# DVERGE training
#python train/train_dverge.py --gpu $GPUID --model-num 8 --distill-eps 0.05 --distill-alpha 0.005 --distill-steps 10 \
#                             --num-class 10 --arch "VggNet" --depth 19

#python train/get_example.py --gpu $GPUID --model-num 5 --distill-eps 0.07 --distill-alpha 0.007 --distill-steps 10

# Baseline training
#python train/train_baseline.py --gpu $GPUID --model-num 8 --num-class 10 --arch "VggNet" --depth 19 --batch-size 128
#python train/train_baseline_data.py --gpu $GPUID --model-num 1 --batch-size 100

# ADP training
#python train/train_adp.py --gpu $GPUID --model-num 5 --num-class 10 --arch "VggNet" --depth 19 --batch-size 128

# GAL training
#python train/train_gal.py --gpu $GPUID --model-num 5 --num-class 10 --arch "VggNet" --depth 19 --batch-size 128

# ADV training
#python train/train_adv.py --gpu $GPUID --model-num 3 --num-class 100

# Trade training
#python train/train_trade.py --gpu $GPUID --model-num 3 --num-class 100

# TRS training
#python train/train_trs.py --gpu $GPUID --model-num 8 --num-class 10 --lr 0.001 --plus-adv --batch-size 128 --arch "VggNet" --depth 19

# LAFED training
python train/train_mixup_dverge.py --gpu $GPUID --model-num 8 --distill-eps 0.07 --distill-alpha 0.007 \
                        --dverge-coeff 1.0 --distill-steps 10 --train-method "mixup" --batch-size 128 --num-class 10 \
                        --soft-label 0.05 \
                        --arch "VggNet" --depth 19

