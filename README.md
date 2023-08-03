# CDM_ILOG
> This is a comprehensive repository specializing in Cognitive Diagnosis Models.

## QCCDM-ECAI2023

This repository has implementation of QCCDM: A Q-Augmented Causal Cognitive Diagnosis Model for Student Learning (**ECAI2023**) which lies in "method" directory.  If you're interested, please navigate to the method/QCCDM directory for more information.

> cd method/QCCDM



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

<u>We utilize **wandb**, a practical and effective package for visualizing our results. However, if you prefer not to use it, it can be easily disabled.</u> https://wandb.ai/



## QCCDM

We provide comprehensive instructions on how to run QCCDM in the "exps/QCCDM" directory. If you're interested, please navigate to the exps/QCCDM directory for more information.

> cd exps/QCCDM



# The File Tree

>
>     .
>     │  .gitattributes
>     │  .gitignore
>     │  LICENSE
>     │  README.md
>     │  requirements.txt
>     │
>     ├─data
>     │  │  data_analysis.py
>     │  │  data_params_dict.py
>     │  │
>     │  ├─junyi
>     │         ground_truth.xlsx
>     │         junyihier.csv
>     │         junyiTotalData.csv
>     │         q.csv
>     │         test_0.8_0.2.csv
>     │         train_0.8_0.2.csv
>     │  
>     │  
>     │          
>     │
>     ├─exps
>     │  └─QCCDM
>     │          exp-cdm-1.sh
>     │          exp-cdm.sh
>     │          exp.py
>     │          README.md
>     │
>     ├─method
>     │  │  __init__.py
>     │  │
>     │  ├─QCCDM
>     │        qccdm.py
>     │        __init__.py
>     │  
>     │  
>     │  
>     │  
>     │  
>     │          
>     │
>     ├─metrics
>     │     DOA.py
>     │     MAD.py
>     │     MLS.py
>     │  
>     │  
>     │          
>     │
>     ├─papers
>     │  └─QCCDM
>     │          Appendix.pdf
>     │          QCCDM_ECAI23.pdf
>     │
>     ├─plot
>     └─runners
>         ├─commonutils
>         │     commonrunners.py
>         │     datautils.py
>         │     util.py
>         │  
>         │  
>         │         
>         │          
>         │
>         └─QCCDM
>                cdm_runners.py
>                utils.py
>     

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
