for i in $(seq 5 9); do
  for mode in train full; do
    for cdm_type in lightgcn ncdm mirt irt; do
      python exp_ind.py --exp_type=ind --method=igcdm --datatype=FrcSub --test_size=0.2 --seed=$i --dim=64 --epoch=8 --device=cuda:0 --gcnlayers=3 --lr=5e-4 --agg_type=mean --mode=$mode --cdm_type=$cdm_type --new_ratio=0.5
      python exp_ind.py  --exp_type=ind --method=igcdm --datatype=nips20 --test_size=0.2 --seed=$i --dim=64 --epoch=5 --device=cuda:0 --gcnlayers=3 --lr=5e-4 --agg_type=mean --mode=$mode --cdm_type=$cdm_type --new_ratio=0.5
      python exp_ind.py  --exp_type=ind --method=igcdm --datatype=a17 --test_size=0.2 --seed=$i --dim=64 --epoch=5 --device=cuda:0 --gcnlayers=1 --lr=1e-3 --agg_type=mean --mode=$mode --cdm_type=$cdm_type --new_ratio=0.5
      done
  done
done