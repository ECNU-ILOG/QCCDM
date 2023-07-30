for i in $(seq 5 9); do
  for cdm_type in mirt irt; do
      python exp.py --exp_type=cdm --method=igcdm --datatype=FrcSub --test_size=0.2 --seed=$i --dim=64 --epoch=8 --device=cuda:2 --gcnlayers=3 --lr=5e-4 --agg_type=mean --cdm_type=$cdm_type
      python exp.py --exp_type=cdm --method=igcdm --datatype=nips20 --test_size=0.2 --seed=$i --dim=64 --epoch=5 --device=cuda:2 --gcnlayers=3 --lr=5e-4 --agg_type=mean --cdm_type=$cdm_type
       python exp.py --exp_type=cdm --method=igcdm --datatype=a17 --test_size=0.2 --seed=$i --dim=64 --epoch=5 --device=cuda:2 --gcnlayers=1 --lr=1e-3 --agg_type=mean --cdm_type=$cdm_type
     done
done