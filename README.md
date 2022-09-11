# Learning-Gaussian-distributions-in-TensorFlow
Template TensorFlow code for feed-forward neural networks - learning Gaussian distributions. <br> <br>
Model architecture illustration: <br> <br>
<img width="487" alt="gaussian_model_architecture_illustration" src="https://user-images.githubusercontent.com/38408538/189530534-7cc8a98d-669c-42d5-8f9e-58e1dce4e69f.png">
<br><br>
- **Input values**: vector of numbers $x$. <br>
- **Output values**: $\mu$, $\sigma$ (parameters of Gaussian distrbution conditioned on input $x$). <br>
- **Loss function**: standard negative log-likelihood of target value $y$ under model output distribution: $-\log p(y; \mu, \sigma)$, where $\mu$, $\sigma = f(x)$ and $f$ is neural network model.

<br> <br> <br>
**Model architecture**: <br>
- feed-forward neural network (model can be extended for recurrent architectures),
- base layers learn joint representations of inputs,
- parameter layers ($\mu$ - layer and $\sigma$ - layer) learn specific representations important for each output parameter.
