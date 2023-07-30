# Implemention of ULCDF: A Unified yet Lightweight Graph-based Cognitive Diagnosis Framework


# Example

Here, we provide an example for running on some datasets. The steps for running the other benchmarks are the same.


Run the code:

## ULCDF with Proposed Interactive Function
```
python exp.py --exp_type=cdm --method=ulcdf --datatype=junyi --test_size=0.2 --epoch=8 --seed=0 --device=cuda:0   --gcnlayers=3 --dim=64 --if_type=ulcdf --lr=5e-4 --weight_reg=1e-3 --keep_prob=0.9  --leaky=0.8  --mode=all
python exp.py --exp_type=cdm --method=ulcdf --datatype=a17 --test_size=0.2 --epoch=8 --seed=0 --device=cuda:0   --gcnlayers=3 --dim=64 --if_type=ulcdf --lr=2e-3 --weight_reg=1e-3 --keep_prob=0.9 --leaky=0.8 --mode=all
python exp.py --exp_type=cdm --method=ulcdf --datatype=nips20 --test_size=0.2 --epoch=8 --seed=0 --device=cuda:0  --gcnlayers=1 --dim=64 --if_type=ulcdf  --lr=5e-4 --weight_reg=1e-3 --keep_prob=0.9   --leaky=0.8 --mode=all
```

## ULCDF with Other Interactive Functions
```
python exp.py --exp_type=cdm --method=ulcdf --datatype=junyi --test_size=0.2 --epoch=8 --seed=0 --device=cuda:0   --gcnlayers=3 --dim=64 --if_type=ncdm  --lr=5e-4 --weight_reg=1e-3 --keep_prob=0.9  --leaky=0.8  --mode=all
python exp.py --exp_type=cdm --method=ulcdf --datatype=junyi --test_size=0.2 --epoch=8 --seed=0 --device=cuda:0   --gcnlayers=3 --dim=64 --if_type=mf --lr=5e-4 --weight_reg=1e-3 --keep_prob=0.9   --leaky=0.8 --mode=all
python exp.py --exp_type=cdm --method=ulcdf --datatype=junyi --test_size=0.2 --epoch=8 --seed=0 --device=cuda:0   --gcnlayers=3 --dim=64 --if_type=mirt --lr=5e-4 --weight_reg=1e-3 --keep_prob=0.9  --leaky=0.8  --mode=all
```
