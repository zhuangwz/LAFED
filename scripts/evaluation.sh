GPUID=5

cd ..

#python eval/save_region_prediction_v1.py \
#    --gpu $GPUID \
#    --model-file checkpoints/fme/seed_0/3_ResNet20/mixup_eps0.07_norm_adplabel0.05_avg_2/epoch_191.pth \
#    --num-class 10

# evaluation of black-box robustness
# remember to first download and put transfer_adv_examples/
# under ../data/


python eval/eval_bbox.py \
    --gpu $GPUID \
    --model-file checkpoints/gal/seed_0/5_VggNet19/epoch_200.pth \
    --folder transfer_adv_examples \
    --steps 100 \
    --num-class 10 \
    --arch "VggNet" --depth 19 \
    --save-to-csv


#python eval/eval_bbox_vertex.py \
#    --gpu $GPUID \
#    --model-file checkpoints/vertex/seed_0/1_ResNet20/best_model_at_cifar100_RN-20_0 \
#    --folder transfer_adv_examples_cifar100 \
#    --steps 100 \
#    --num-class 100 \
#    --save-to-csv

# evaluation of white-box robustness
python eval/eval_wbox.py \
    --gpu $GPUID \
    --model-file checkpoints/gal/seed_0/5_VggNet19/epoch_200.pth \
    --steps 50 \
    --random-start 5 \
    --num-class 10 \
    --arch "VggNet" --depth 19 \
    --save-to-csv

#python eval/eval_wbox_vertex.py \
#  --gpu $GPUID \
#  --model-file checkpoints/vertex/seed_0/1_ResNet20/best_model_at_cifar100_RN-20_0 \
#  --steps 50 \
#  --random-start 5 \
#  --num-class 100 \
#  --save-to-csv

#python eval/eval_wbox_aa.py \
#    --gpu $GPUID \
#    --model-file checkpoints/trs/seed_0/5_ResNet20_eps_0.01_0.03_adam_plus_adv_coeff_20.0_at/epoch_200.pth \
#    --steps 50 \
#    --random-start 1 \
#    --num-class 100 \
#    --save-to-csv

# evaluation of transferability
#python eval/eval_transferability.py \
#    --gpu $GPUID \
#    --model-file checkpoints/fme/seed_0/8_ResNet20/mixup_eps0.07_norm_adplabel0.05_avg/epoch_192.pth \
#    --steps 50 \
#    --random-start 5 \
#    --num-class 10 \
#    --save-to-file

# evaluation of diversity
#python eval/eval_diversity.py \
#    --gpu $GPUID \
#    --model-file checkpoints/dverge/seed_0/5_ResNet20_eps_0.05/epoch_200.pth \
#    --save-to-file

# my
#python eval/eval_wbox.py \
#    --gpu $GPUID \
#    --model-file checkpoints/gal/seed_0/8_ResNet20_plus_adv/epoch_200.pth \
#    --steps 50 \
#    --random-start 5 \
#    --save-to-csv

#python eval/eval_KL.py \
#    --gpu $GPUID \
#    --model-file checkpoints/my/seed_0/3_ResNet20/mixup0.3_dverge_eps0.07/epoch_200.pth \
#    --save-to-file

#python eval/eval_epoch.py \
#    --gpu $GPUID \
#    --model-file checkpoints/fme/seed_0/5_ResNet20/dverge_eps0.07_norm_adplabel0.05_avg \
#    --steps 50 \
#    --random-start 5 \
#    --num-class 10 \
#    --save-to-csv

#python eval/eval_feature_distance.py \
#    --gpu $GPUID \
#    --model-file checkpoints/my/seed_0/5_ResNet20/mixup1.00_dverge_eps0.07/epoch_200.pth \
#    --batch-size 1 \
#    --save-to-csv

#python eval/eval_feature_cos.py \
#    --gpu $GPUID \
#    --model-file checkpoints/fme/seed_0/5_ResNet20/mixup_eps0.07_norm_adplabel0.05_avg/epoch_198.pth \
#    --save-to-csv

#python eval/eval_feature_weight.py \
#    --gpu $GPUID \
#    --model-file checkpoints/my1/seed_0/3_ResNet20/dverge_multi_0.07/epoch_200.pth

#python eval/eval_combine_two.py \
#    --gpu $GPUID \
#    --model-file checkpoints/baseline_data/seed_0 \
#    --steps 50 \
#    --random-start 5 \
#    --save-to-file

#python eval/generate_bbox_sim.py \
#    --gpu $GPUID \
#    --model-file checkpoints/baseline/seed_0/3_ResNet20/epoch_200.pth \
#    --num-class 10 \
#    --num-steps 100 \
#    --gamma 0.2 \
#    --cw-conf 100.0 \
#    --momentum 1.0 \
#    --mdi-prob 0.0 \
#    --random-start 1 \
#    --output-dir transfer_adv_examples_sa_conv3_5

#python eval/eval_pcc.py \
#    --gpu $GPUID \
#    --save-to-csv

#python eval/eval_feature_pcc.py \
#    --gpu $GPUID \
#    --save-to-csv

#python eval/eval_baseline_pcc.py \
#    --gpu $GPUID \
#    --save-to-csv
