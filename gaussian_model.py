from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions as tfd
import tensorflow as tf

def build_gaussian_model(input_dim, base_layers = [64], mu_layers = [64], 
                sigma_layers = [64], optimizer = Adam(), batch_normalization = False, 
                dropout_rate = None, regularization = None, hidden_activation = 'elu', 
                mu_output_activation = 'linear', sigma_output_activation = 'softplus', 
                model_metrics = None, model_name = 'gaussian_model'):
    assert base_layers, 'base_layers argument is empty'

    # arguments base_layers, mu_layers, sigma_layers are lists of 
    # neurons for each of the layers: mu_layers and sigma_layers can be empty lists

    # TF-Keras Functional API for building models
    model_input = Input(shape = (input_dim,), name = 'input_layer')

    # add hidden base layers (possibility of adding dropout and batch norm)
    layer = Dense(base_layers[0], activation = hidden_activation, name = 'base_layer_1', 
                      kernel_regularizer = regularization)(model_input)
    if len(base_layers) > 1:
        for ind in range(1, len(base_layers)):
            if batch_normalization:
                layer = BatchNormalization()(layer)
            if dropout_rate is not None and dropout_rate > 0:
                layer = Dropout(dropout_rate)(layer)
            layer = Dense(base_layers[ind], activation = hidden_activation, name = 'base_layer_{}'.format(ind+1),
                          kernel_regularizer = regularization)(layer)

    # mu layers branch (possibility of adding dropout and batch norm)
    mu_layer = layer
    if mu_layers is not None and len(mu_layers) > 0:
        for k in range(len(mu_layers)):
            mu_layer = Dense(units = mu_layers[k], activation = hidden_activation)(mu_layer)
            if batch_normalization:
                mu_layer = BatchNormalization()(mu_layer)
            if dropout_rate is not None and dropout_rate > 0:
                mu_layer = Dropout(rate = dropout_rate)(mu_layer)
    mu_output = Dense(units = 1, activation = mu_output_activation, name = 'mu_output_layer',
                      kernel_regularizer = regularization)(mu_layer)

    # sigma layers branch (possibility of adding dropout and batch norm)
    sigma_layer = layer
    if sigma_layers is not None and len(sigma_layers) > 0:
        for z in range(len(sigma_layers)):
            sigma_layer = Dense(units = sigma_layers[z], activation = hidden_activation)(sigma_layer)
            if batch_normalization:
                sigma_layer = BatchNormalization()(sigma_layer)
            if dropout_rate is not None and dropout_rate > 0:
                sigma_layer = Dropout(rate = dropout_rate)(sigma_layer)
    sigma_output = Dense(units = 1, activation = sigma_output_activation, name = 'sigma_output_layer',
                      kernel_regularizer = regularization)(sigma_layer)

    # output layer: concatenation of two branches for mu and sigma parameters
    model_output = Concatenate(name = 'output_layer')([mu_output, sigma_output])
    model = Model(model_input, model_output, name = model_name)

    @tf.function
    def loss_function(y_true, y_pred):
        '''loss function: negative log-likelihood of target values under 
        model output distribution,
        y_true: true labels (target values),
        y_pred: predicted parameters of gaussian distribution'''
        # model predictions: mu and sigma
        mu, sigma = tf.unstack(y_pred, num = 2, axis = -1)
        mu = tf.expand_dims(mu, axis = -1)
        sigma = tf.expand_dims(sigma, axis = -1)
        # gaussian distribution with parameters 'mu' and 'sigma'
        gaussian_dist = tfd.Normal(loc = mu, scale = sigma)
        # loss: mean negative log-likelihood values
        return tf.reduce_mean(-gaussian_dist.log_prob(y_true))

    # training configuration: loss function, optimizer and metrics (if any)
    model.compile(optimizer = optimizer, loss = loss_function, metrics = model_metrics)

    return model
    