# Process
> 2023/7/14 Considering for Inductive Graph Cognitive Diagnosis Model (status: Fail)
> 2023/7/27 Considering for Inductive Graph Cognitive Diagnosis Model (status: Half Success) in FrcSub nips20 a17

```
python exp.py --exp_type=cdm --method=igcdm --datatype=FrcSub --test_size=0.2 --seed=1 --dim=64 --epoch=8 --device=cuda --gcnlayers=3 --lr=5e-4 --agg_type=mean
```

# Best Hyper-Parameters
| Dataset | #Dim | #GCNlayers | #lr  | 
|---------|------|------------|------|
| FrcSub  | 64   | 3          | 5e-4 | 
| nips20  | 64   | 3          | 5e-4 | 
| a17     | 64   | 1          | 1e-3 | 