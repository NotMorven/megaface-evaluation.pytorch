#!/usr/bin/env bash

ROOT="./"
OUTDIR="/home/morven/MorvenWorkspace/0mega_feature_out"
echo $ROOT
DEVKIT="$ROOT/devkit/devkit/experiments"

#################################################################################
ALGO="AMSoftmax-2020-11-30-03-16"
echo $ALGO
MODEL="/home/morven/MorvenWorkspace/UltiLMC/checkpoints/2020-11-30-03-16_AddCosine_s_18_m_0p35/2020-11-30-03-16_AddCosine_s_18_m_0p35_stateDict_epochs_18_best_weight_epoch_17_acc_0p9914999999999999.pkl"

python -u gen_megaface.py --gpu 0 --algo "$ALGO" --model "$MODEL" --facescrub-lst "$ROOT/data/megaface_testpack/facescrub_lst" --megaface-lst "$ROOT/data/megaface_testpack/megaface_lst" --facescrub-root "$ROOT/data/megaface_testpack/facescrub_images" --megaface-root "$ROOT/data/megaface_testpack/megaface_images" --output "$OUTDIR/"$ALGO"_feature_out"

python -u remove_noises.py --algo "$ALGO" --facescrub-noises "$ROOT/data/megaface_testpack/facescrub_noises.txt" --megaface-noises "$ROOT/data/megaface_testpack/megaface_noises.txt" --facescrub-lst "$ROOT/data/megaface_testpack/facescrub_lst" --megaface-lst "$ROOT/data/megaface_testpack/megaface_lst" --feature-dir-input "$OUTDIR/"$ALGO"_feature_out" --feature-dir-out "$OUTDIR/"$ALGO"_feature_out_clean"

#export LD_LIBRARY_PATH="/usr/local/lib/opencv2.4:$LD_LIBRARY_PATH"
#python -u devkit/experiments/run_experiment.py "$ROOT/"$ALGO"_feature_out_clean/megaface" "$ROOT/"$ALGO"_feature_out_clean/facescrub" _"$ALGO".bin ./"$ALGO"_results/ -p ../templatelists/facescrub_features_list.json

#################################################################################
ALGO="AMSoftmax-2020-12-18-22-29"
echo $ALGO
MODEL="/home/morven/MorvenWorkspace/UltiLMC/checkpoints/2020-12-18-22-29_AddCosine_s_30_m_0p35_lr_0p1_IntraLoss_alpha_10p0_beta_18p8_warm_up_1/2020-12-18-22-29_AddCosine_s_30_m_0p35_lr_0p1_IntraLoss_alpha_10p0_beta_18p8_warm_up_1_stateDict_epochs_25_best_weight_epoch_13_acc_0p9916666666666668.pkl"

python -u gen_megaface.py --gpu 0 --algo "$ALGO" --model "$MODEL" --facescrub-lst "$ROOT/data/megaface_testpack/facescrub_lst" --megaface-lst "$ROOT/data/megaface_testpack/megaface_lst" --facescrub-root "$ROOT/data/megaface_testpack/facescrub_images" --megaface-root "$ROOT/data/megaface_testpack/megaface_images" --output "$OUTDIR/"$ALGO"_feature_out"

python -u remove_noises.py --algo "$ALGO" --facescrub-noises "$ROOT/data/megaface_testpack/facescrub_noises.txt" --megaface-noises "$ROOT/data/megaface_testpack/megaface_noises.txt" --facescrub-lst "$ROOT/data/megaface_testpack/facescrub_lst" --megaface-lst "$ROOT/data/megaface_testpack/megaface_lst" --feature-dir-input "$OUTDIR/"$ALGO"_feature_out" --feature-dir-out "$OUTDIR/"$ALGO"_feature_out_clean"

#export LD_LIBRARY_PATH="/usr/local/lib/opencv2.4:$LD_LIBRARY_PATH"
#python -u devkit/experiments/run_experiment.py "$ROOT/"$ALGO"_feature_out_clean/megaface" "$ROOT/"$ALGO"_feature_out_clean/facescrub" _"$ALGO".bin ./"$ALGO"_results/ -p ../templatelists/facescrub_features_list.json

#################################################################################
ALGO="AMSoftmax-2020-12-19-03-25"
echo $ALGO
MODEL="/home/morven/MorvenWorkspace/UltiLMC/checkpoints/2020-12-19-03-25_AddCosine_s_30_m_0p35_lr_0p1_IntraLoss_alpha_10p0_beta_18p8_warm_up_1/2020-12-19-03-25_AddCosine_s_30_m_0p35_lr_0p1_IntraLoss_alpha_10p0_beta_18p8_warm_up_1_stateDict_epochs_25_best_weight_epoch_10_acc_0p9913333333333334.pkl"

python -u gen_megaface.py --gpu 0 --algo "$ALGO" --model "$MODEL" --facescrub-lst "$ROOT/data/megaface_testpack/facescrub_lst" --megaface-lst "$ROOT/data/megaface_testpack/megaface_lst" --facescrub-root "$ROOT/data/megaface_testpack/facescrub_images" --megaface-root "$ROOT/data/megaface_testpack/megaface_images" --output "$OUTDIR/"$ALGO"_feature_out"

python -u remove_noises.py --algo "$ALGO" --facescrub-noises "$ROOT/data/megaface_testpack/facescrub_noises.txt" --megaface-noises "$ROOT/data/megaface_testpack/megaface_noises.txt" --facescrub-lst "$ROOT/data/megaface_testpack/facescrub_lst" --megaface-lst "$ROOT/data/megaface_testpack/megaface_lst" --feature-dir-input "$OUTDIR/"$ALGO"_feature_out" --feature-dir-out "$OUTDIR/"$ALGO"_feature_out_clean"

#export LD_LIBRARY_PATH="/usr/local/lib/opencv2.4:$LD_LIBRARY_PATH"
#python -u devkit/experiments/run_experiment.py "$ROOT/"$ALGO"_feature_out_clean/megaface" "$ROOT/"$ALGO"_feature_out_clean/facescrub" _"$ALGO".bin ./"$ALGO"_results/ -p ../templatelists/facescrub_features_list.json

##################################################################################
ALGO="AMSoftmax-Intral-2020-12-18-17-32"
echo $ALGO
MODEL="/home/morven/MorvenWorkspace/UltiLMC/checkpoints/2020-12-18-17-32_AddCosine_s_30_m_0p35_lr_0p1_IntraLoss_alpha_10p0_beta_18p8_warm_up_1/2020-12-18-17-32_AddCosine_s_30_m_0p35_lr_0p1_IntraLoss_alpha_10p0_beta_18p8_warm_up_1_stateDict_epochs_25_best_weight_epoch_13_acc_0p9928333333333332.pkl"

python -u gen_megaface.py --gpu 0 --algo "$ALGO" --model "$MODEL" --facescrub-lst "$ROOT/data/megaface_testpack/facescrub_lst" --megaface-lst "$ROOT/data/megaface_testpack/megaface_lst" --facescrub-root "$ROOT/data/megaface_testpack/facescrub_images" --megaface-root "$ROOT/data/megaface_testpack/megaface_images" --output "$OUTDIR/"$ALGO"_feature_out"

python -u remove_noises.py --algo "$ALGO" --facescrub-noises "$ROOT/data/megaface_testpack/facescrub_noises.txt" --megaface-noises "$ROOT/data/megaface_testpack/megaface_noises.txt" --facescrub-lst "$ROOT/data/megaface_testpack/facescrub_lst" --megaface-lst "$ROOT/data/megaface_testpack/megaface_lst" --feature-dir-input "$OUTDIR/"$ALGO"_feature_out" --feature-dir-out "$OUTDIR/"$ALGO"_feature_out_clean"

#export LD_LIBRARY_PATH="/usr/local/lib/opencv2.4:$LD_LIBRARY_PATH"
#python -u devkit/experiments/run_experiment.py "$ROOT/"$ALGO"_feature_out_clean/megaface" "$ROOT/"$ALGO"_feature_out_clean/facescrub" _"$ALGO".bin ./"$ALGO"_results/ -p ../templatelists/facescrub_features_list.json

