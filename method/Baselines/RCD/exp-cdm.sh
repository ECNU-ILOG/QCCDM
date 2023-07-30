for i in $(seq 0 9); do
  python main.py --exp_type=cdm  --datatype=nips20 --seed=$i --device=cuda:0 --epoch=5 --test_size=0.2
#  python main.py --exp_type=cdm --datatype=junyi --seed=$i --device=cuda:0 --epoch=5 --test_size=0.2
#    python main.py --exp_type=cdm --datatype=a910 --seed=$i --device=cuda:0 --epoch=5 --test_size=0.2
#  python main.py --exp_type=cdm  --datatype=a17 --seed=$i --device=cuda:0 --epoch=5 --test_size=0.2
done
