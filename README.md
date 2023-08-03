# CDM_ILOG
> This is a comprehensive repository specializing in Cognitive Diagnosis Models.

We support our QCCDM which lies in the "method" directory.

# Requirements

```python
EduCDM==0.0.13
joblib==1.2.0
numpy==1.23.5
pandas==1.5.3
scikit_learn==1.2.2
torch==1.13.1+cu117
tqdm==4.65.0
wandb==0.15.2
```

# Experiment





# The File Tree

>
>     .  
>     |   LICENSE
>     |   output.txt
>     |   README.md
>     |   
>     +---data
>     |   |   data_analysis.py
>     |   |   data_params_dict.py
>     |   |   
>     |   +---junyi
>     |   |       ground_truth.xlsx
>     |   |       junyihier.csv
>     |   |       junyiTotalData.csv
>     |   |       q.csv
>     |   |       test_0.8_0.2.csv
>     |   |       train_0.8_0.2.csv
>     |         
>     |   
>     |           
>     |           
>     +---exps
>     |   \---QCCDM
>     |           exp-cdm-1.sh
>     |           exp-cdm.sh
>     |           exp.py
>     |           README.md
>     |           
>     +---method
>     |   |   __init__.py
>     |   |   
>     |   +---QCCDM
>     |       |   qccdm.py
>     |       |   __init__.py
>     |   
>     |   
>     |   
>     |   
>     |   
>     |          
>     |           
>     +---metrics
>     |   |   DOA.py
>     |   |   MAD.py
>     |   |   MLS.py
>     |  
>     |   
>     |           
>     |           
>     +---plot
>     \---runners
>         +---commonutils
>         |   |   commonrunners.py
>         |   |   datautils.py
>         |   |   util.py
>         |   
>         |   
>         |           
>         |           
>         |           
>         \---QCCDM
>             |   cdm_runners.py
>             |   utils.py          

# Reference

Shuo Liu, Hong Qian, Mingjia Li, Aimin Zhou "QCCDM: A Q-Augmented Causal Cognitive Diagnosis Model for Student Learning." In Proceedings of the 26th European Conference on Artificial Intelligence, 2023.

## Bibtex

> ```
> @inproceedings{liu2023ecai,
> author = {Shuo Liu, Hong Qian, Mingjia Li, Aimin Zhou},
> booktitle = {Proceedings of the 26th European Conference on Artificial Intelligence},
> title = {QCCDM: A Q-Augmented Causal Cognitive Diagnosis Model for Student Learning},
> year = {2023},
> address={Kraków, Poland}
> }
> ```
