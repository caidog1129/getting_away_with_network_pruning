Use mp.cpu_count() to determine how many cpu you have on your machine, change line 36 in prob.py to that number. Change sample factor in line 45 in ub.py to be the closet integer to 50 that divides number of cpus. For example, have 12 cpus, change it to 48. 

1. Use Find_sparsity.ipynb to find the appropriate value for global sparsity. Choose 7 global sparsity rate exclude 1. 
2. Run ub.py, #Usage: python ub.py --n "[num_samples-1,100,100,10]" --p "[0.01, 0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.0001]" --name "MNIST_s[num_samples]", it will save a file called name+_ub.csv
3. Run ub_uniform.py, #Usage: python ub_uniform.py --n "[num_samples-1,100,100,10]" --p "[replace with 7 global sparsity]" --name "MNIST_s[num_samples]", it will save a file called name+_ub_u.csv
4. run check_exact.py: For example, for Perceptron100 with num_samples = 3: #Usage: check_exact.py --n "[784,100,100,10]" --p "[0.01, 0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.0001]" --name "MNIST_s3" --model "Perceptron100" --dataset "MNIST" --sample_points 3
