for i in $(seq 0 9); do
#  python exp.py --exp_type=cdm --method=rcd --datatype=FrcSub --test_size=0.2 --seed=$i --device=cuda:0 --epoch=15
#  python exp.py --exp_type=cdm --method=rcd --datatype=Math1 --test_size=0.2 --seed=$i --device=cuda:0 --epoch=5
#  python exp.py --exp_type=cdm --method=rcd --datatype=Math2 --test_size=0.2 --seed=$i --device=cuda:0 --epoch=5
  python exp.py --exp_type=cdm --method=rcd --datatype=a17 --test_size=0.2 --seed=$i --device=cuda:0 --epoch=5
#  python exp.py --exp_type=cdm --method=kancd --datatype=a17 --test_size=0.2 --seed=0 --device=cuda:0 --epoch=5
done