# Poison-adv-training
Poisoning attack methods against adversarial training algorithms.

## Usage
### Targeted Attack
You can run the following command to generate clean-label targeted poison data and validate it on the robust learner.
```
python tar_poisoning.py --poisonkey 2000000000 --eps 16 --tau 0.01 --budget 0.04 --attackiter 250 --restarts 8 --vruns 4
```

### Untargeted Attack
Run the following command to generate clean-label untargeted poison data.
```
python untar_poisoning.py --train-steps 5000 --optim sgd --lr 0.1 --lr-decay-rate 0.1 --lr-decay-freq 2000 --pgd-steps 10 --pgd-step-size 35 --pgd-random-start --patch-size 0.03 --location 4
```
Then, run the following command to validate clean-label untargeted poison data on the robust learner.
```
python validate_untar_poisoning.py --seed 2000000000 --noise-rate 0.5 --train-steps 15000 --optim sgd --lr 0.1 --lr-decay-rate 0.1 --lr-decay-freq 6000 --pgd-radius 8 --pgd-steps 10 --pgd-step-size 1.6 --pgd-random-start --report-freq 200 --save-freq 100000 --noise-path ./exp_data/untargeted/patch_cifar10_loc4_ss35_ps0.03/poisons/patch-fin-def-noise.pkl --mask-path ./exp_data/untargeted/patch_cifar10_loc4_ss35_ps0.03/poisons/patch-fin-def-mask.pkl --save-name train
```
