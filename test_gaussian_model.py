import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.random import set_seed
import numpy as np
from gaussian_model import build_gaussian_model
plt.rcParams["figure.figsize"] = [16,8]
sns.set_style("darkgrid")

np.random.seed(1)
# first - create toy dataset: inputs x and target variables y
# input values are vector of size 5, target values are numbers 
# like in regression models
x = np.random.uniform(-1, 1, size = (500, 5))
y = np.random.uniform(-5, 5, size = 500)

# define model
np.random.seed(1)
set_seed(1)
model = build_gaussian_model(input_dim = x.shape[-1], base_layers = [32], mu_layers = [], 
                sigma_layers = [], batch_normalization = False, dropout_rate = None,
                regularization = None, hidden_activation = 'elu', mu_output_activation = 'linear', 
                sigma_output_activation = 'softplus')
model.summary()

# model training and hyperparameters
np.random.seed(1)
set_seed(1)
EPOCHS = 100
BATCH_SIZE = 512
training_history = model.fit(x, y, epochs = EPOCHS, batch_size = BATCH_SIZE, 
                             shuffle = True, verbose = 1)

# plot training results
epoch = list(range(1, len(training_history.history['loss']) + 1))
plt.plot(epoch, training_history.history['loss'], '-', label='train', linewidth=3, color= 'blue')
plt.xticks(fontsize=22)
plt.yticks(fontsize=22);
plt.xlabel('Epoch [-]', fontsize=20);
plt.ylabel('Loss: negative log-likelihood [-]', fontsize=20)
plt.legend(prop={'size': 20});