for i in $(seq 5 9); do
  for nonlinear in sigmoid softplus tanh; do
    for q_aug in single mf; do
      python exp.py --exp_type=cdm --method=qccdm --datatype=junyi --test_size=0.2 --seed=$i --num_layers=3 --nonlinear=$nonlinear --q_aug=$q_aug --batch_size=128 --device=cuda:0 --epoch=5
      python exp.py --exp_type=cdm --method=qccdm --datatype=Math1 --test_size=0.2 --seed=$i --num_layers=3 --nonlinear=$nonlinear --q_aug=$q_aug --device=cuda:0 --epoch=5
      python exp.py --exp_type=cdm --method=qccdm --datatype=Math2 --test_size=0.2 --seed=$i --num_layers=2 --nonlinear=$nonlinear --q_aug=$q_aug --device=cuda:0 --epoch=5
      python exp.py --exp_type=cdm --method=kancd --datatype=junyi --test_size=0.2 --seed=$i --device=cuda:0 --batch_size=128
    python exp.py --exp_type=cdm --method=kancd --datatype=Math1 --test_size=0.2 --seed=$i --device=cuda:0
    python exp.py --exp_type=cdm --method=kancd --datatype=Math2 --test_size=0.2 --seed=$i --device=cuda:0
      python exp.py --exp_type=cdm --method=ncdm --datatype=junyi --test_size=0.2 --seed=$i --device=cuda:0 --batch_size=128
    python exp.py --exp_type=cdm --method=ncdm  --datatype=Math1 --test_size=0.2 --seed=$i --device=cuda:0
    python exp.py --exp_type=cdm --method=ncdm  --datatype=Math2 --test_size=0.2 --seed=$i --device=cuda:0
      done
  done
done