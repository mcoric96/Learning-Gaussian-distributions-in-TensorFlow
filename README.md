# Learning-Gaussian-distributions-in-TensorFlow
Template TensorFlow code for feed-forward neural networks - learning Gaussian distributions. <br> <br>
Model architecture illustration: <br> <br>
<img width="487" alt="gaussian_model_architecture_illustration" src="https://user-images.githubusercontent.com/38408538/189530534-7cc8a98d-669c-42d5-8f9e-58e1dce4e69f.png">
<br><br>
- **Input values**: vector of numbers $x$. <br>
- **Output values**: $\mu$, $\sigma$ (parameters of Gaussian distrbution conditioned on input $x$). <br>
- **Loss function**: standard negative log-likelihood of target value $y$ under model output distribution: $-\log p(y; \mu, \sigma)$, where $\mu$, $\sigma = f(x)$ and $f$ is neural network model.
<br>
For dataset $X = \{x_1, x_2, ..., x_n\}$ and target values $Y = \{y_1, y_2, ..., y_3\}$ loss function negative log-likelihood is defined as:
<p align="center">
$$L(\theta ; X) = \frac{1}{n} \sum_{i=1}^n -\log p(y_i; \mu_i, \sigma_i)$$
</p>
where $\mu_i$ and $\sigma_i$ are model outputs for $x_i$, $\theta$ are model parameters and $p$ is probability density function defined with model outputs $\mu_i$ and $\sigma_i$.
<br>

<br> <br>
**Model architecture**: <br>
- feed-forward neural network (model can be extended for recurrent architectures),
- base layers learn joint representations of inputs,
- parameter layers ( $\mu$ - layer and $\sigma$ - layer) learn specific representations important for each output parameter,
- alternative for regression models with single numerical output,
- $\mu$ - layer output activation function: can be any function, *linear* or any other that restricts output range,
- $\sigma$ - layer output activation function: should be function with only positive value outputs, like *softplus*.

<br> <br>
Model output $\sigma$ represents aleatoric model uncertainty (illustration example below). <br> <br>
<img width="549" alt="github_aleatoric_uncertainty" src="https://user-images.githubusercontent.com/38408538/189534678-69006e78-4abe-4719-b09b-61cea892c5d0.png">

