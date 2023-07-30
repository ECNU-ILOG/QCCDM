# Implemention of SSCDM: A Self-Supervised Cognitive Diagnosis Model with Right-Wrong Response Decomposition for Student Learning


# Example

Here, we provide an example for running Junyi. The steps for running the other benchmarks are the same.


Run the code:

## Without Contrastive Learning
```
python exp.py --exp_type=cdm --method=sscdm --datatype=junyi --test_size=0.2 --epoch=2 --seed=0 --device=cuda:0   --gcnlayers=3 --dim=64

```
## With Contrastive Learning
```
python exp.py --exp_type=cdm --method=sscdm-ssl-ed --datatype=junyi --test_size=0.2 --epoch=2 --seed=0 --device=cuda:0   --gcnlayers=3 --dim=64 --ssl --ratio=0.1  --temp=0.5 --weight_ssl=0.02  --weight_reg=0.15
python exp.py --exp_type=cdm --method=sscdm-ssl-nd --datatype=junyi --test_size=0.2 --epoch=2 --seed=0 --device=cuda:0 --gcnlayers=3 --dim=64 --ssl --ratio=0.1  --temp=0.5 --weight_ssl=0.02  --weight_reg=0.15
```
