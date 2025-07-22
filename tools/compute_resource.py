batch_size = 116
iterations = 30000
total_data = 25000000

samples_per_iteration = batch_size * iterations
epochs = total_data / samples_per_iteration

print("训练的epoch数为:", epochs)