#!/bin/bash
cd ..

#for mask_att_l1 in `seq $1 0.5 $2`
#do
#    echo "Attacking with mask_att_l1 being $mask_att_l1"
#    python ./blurattack.py -d dev -g 1 -m mbAdv_mifgsm -i 1 -w inceptionv3 -s 10 -n -3 -b bg_obj_att -e 0.4,15.0 -a $mask_att_l1
#done
mask_att_l1=0.2
echo "Attacking with mask_att_l1 being $mask_att_l1"
python ./blurattack.py -d dev -g 1 -m mbAdv_mifgsm -i 1 -w inceptionv3 -s 10 -n -3 -b bg_obj_att -e 0.4,15.0 -a $mask_att_l1
python ./blurattack.py -d dev -g 1 -m mbAdv_mifgsm -i 1 -w inceptionresnetv2 -s 10 -n -3 -b bg_obj_att -e 0.4,15.0 -a $mask_att_l1
python ./blurattack.py -d dev -g 1 -m mbAdv_mifgsm -i 1 -w inceptionv4 -s 10 -n -3 -b bg_obj_att -e 0.4,15.0 -a $mask_att_l1
python ./blurattack.py -d dev -g 1 -m mbAdv_mifgsm -i 1 -w xception -s 10 -n -3 -b bg_obj_att -e 0.4,15.0 -a $mask_att_l1

cd experiments

