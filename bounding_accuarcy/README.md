Determine how many cpu you want to use, change line 36 in prob.py to that number. Change sample factor in line 53 in all.py to be the closet integer to 50 that divides number of cpus. For example, have 12 cpus, change it to 48.

python sampling.py/binomial_formula.py --n "[784,100,100,10]" --p "[0.01, 0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.0001]" --name "mnist100" --model "Perceptron100" --dataset "MNIST"
python sampling.py/binomial_formula.py --n "[784,200,200,10]" --p "[0.01, 0.001, 0.0006, 0.0004, 0.0002, 0.0001, 0.00005]" --name "mnist200" --model "Perceptron200" --dataset "MNIST"
python sampling.py/binomial_formula.py --n "[784,400,400,10]" --p "[0.001, 0.0005, 0.00025, 0.0001, 0.000075, 0.00005, 0.000025, 0.00001]" --name "mnist400" --model "Perceptron400" --dataset "MNIST"
python sampling.py/binomial_formula.py --n "[400,120,84,10]" --p "[0.001, 0.00075, 0.000625, 0.0005, 0.000375, 0.00025, 0.0001, 0.00009, 0.00008]" --name "mnistlenet" --model "LeNet" --dataset "MNIST"

python sampling.py/binomial_formula.py --n "[784,100,100,10]" --p "[0.01, 0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.0001]" --name "fashion100" --model "Perceptron100" --dataset "Fashion"
python sampling.py/binomial_formula.py --n "[784,200,200,10]" --p "[0.01, 0.001, 0.0006, 0.0004, 0.0002, 0.0001, 0.00005]" --name "fashion200" --model "Perceptron200" --dataset "Fashion"
python sampling.py/binomial_formula.py --n "[784,400,400,10]" --p "[0.001, 0.0005, 0.00025, 0.0001, 0.000075, 0.00005, 0.000025, 0.00001]" --name "fashion400" --model "Perceptron400" --dataset "Fashion"
python sampling.py/binomial_formula.py --n "[400,120,84,10]" --p "[0.001, 0.00075, 0.000625, 0.0005, 0.000375, 0.00025, 0.0001, 0.00009, 0.00008]"  --name "fashionlenet" --model "LeNet" --dataset "Fashion"

python sampling.py/binomial_formula.py --n "[3072,100,100,10]" --p "[0.01, 0.005, 0.001, 0.00075, 0.0005, 0.00025, 0.0001]" --name "cifar100" --model "Perceptron100_C10" --dataset "CIFAR10"
python sampling.py/binomial_formula.py --n "[3072,200,200,10]" --p "[0.005, 0.001, 0.00075, 0.0005, 0.00025, 0.0001, 0.00005]" --name "cifar200" --model "Perceptron200_C10" --dataset "CIFAR10"
python sampling.py/binomial_formula.py --n "[3072,400,400,10]" --p "[0.001, 0.0005, 0.00025, 0.0001, 0.000075, 0.00005, 0.00001]" --name "cifar400" --model "Perceptron400_C10" --dataset "CIFAR10"
python sampling.py/binomial_formula.py --n "[400,120,84,10]" --p "[0.001, 0.00075, 0.000625, 0.0005, 0.000375, 0.00025, 0.0001, 0.00009, 0.00008]"  --name "cifarlenet" --model "LeNet_C10" --dataset "CIFAR10"
