# Implemention of QCCDM: A Q-Augmented Causal Cognitive Diagnosis Model for Student Learning (ECAI-2023)


# Example

Here, we provide an example for running Junyi. The steps for running the other benchmarks are the same.

```
cd exps
```
Run the code:
```
python exp.py --exp_type=cdm --method=qccdm --datatype=junyi --test_size=0.2 --seed=1 --num_layers=3 --nonlinear=sigmoid --q_aug=mf
python exp.py --exp_type=cdm --method=qccdm --datatype=Math1 --test_size=0.2 --seed=1 --num_layers=3 --nonlinear=sigmoid --q_aug=mf
python exp.py --exp_type=cdm --method=qccdm --datatype=Math2 --test_size=0.2 --seed=1 --num_layers=2 --nonlinear=sigmoid --q_aug=mf
```
## Ablation Study
qccdm only utilize Structure Causal Model
```
python exp.py --exp_type=cdm --method=qccdm-c --datatype=junyi --test_size=0.2 --seed=1 --num_layers=3 --nonlinear=sigmoid --q_aug=mf
```
qccdm only utilize Q-augmentation
```
python exp.py --exp_type=cdm --method=qccdm-q --datatype=junyi --test_size=0.2 --seed=1 --num_layers=3  --nonlinear=sigmoid --q_aug=mf
```