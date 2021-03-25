#!/usr/bin/env bash

ROOT="./"
OUTDIR="/home/morven/MorvenWorkspace/0mega_feature_out"
echo $ROOT
DEVKIT="$ROOT/devkit/devkit/experiments"

export LD_LIBRARY_PATH="/home/morven/.conda/envs/pytensorrt27/lib/opencv2.4:$LD_LIBRARY_PATH"
#ALGO="SOFTMAX"
#python -u devkit/devkit/experiments/run_experiment.py "$OUTDIR/"$ALGO"_feature_out_clean/megaface" "$OUTDIR/"$ALGO"_feature_out_clean/facescrub" _"$ALGO".bin /home/morven/MorvenWorkspace/megaface-evaluation.pytorch/res/"$ALGO"_results/ -p /home/morven/MorvenWorkspace/megaface-evaluation.pytorch/devkit/devkit/templatelists/facescrub_features_list.json
##../templatelists/facescrub_features_list.json

#ALGO="NormFace-2020-12-08-23-36"
#ALGO="NormFace-Intra-2021-01-10-02-19"
#ALGO="MASoftmax-2020-12-27-23-09"
#ALGO="MASoftmax-Intra-server-2020-12-24-03-41"
ALGO="MASoftmax-Intra-2020-12-24-22-25"
echo $ALGO
python -u devkit/devkit/experiments/run_experiment.py "$OUTDIR/"$ALGO"_feature_out_clean/megaface" "$OUTDIR/"$ALGO"_feature_out_clean/facescrub" _"$ALGO".bin /home/morven/MorvenWorkspace/megaface-evaluation.pytorch/res/"$ALGO"_results/ -p /home/morven/MorvenWorkspace/megaface-evaluation.pytorch/devkit/devkit/templatelists/facescrub_features_list.json
#../templatelists/facescrub_features_list.json