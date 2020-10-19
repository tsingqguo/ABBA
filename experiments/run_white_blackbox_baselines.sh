#!/usr/bin/env bash
# bbmodel_names = ["inceptionresnetv2","inceptionv3","inceptionv4","xception"]
bash white_blackbox_baselines.sh fgsm test $1 $2
bash white_blackbox_baselines.sh mifgsm test $1 $2
bash white_blackbox_baselines.sh gblur test $1 $2
bash white_blackbox_baselines.sh dblur test $1 $2