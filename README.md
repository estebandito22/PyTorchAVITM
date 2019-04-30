# PyTorchAVITM
PyTorch Implementation of Autoencoding Variational Inference for Topic Models (Srivastava and Sutton 2017)  


# Why PyTorchAVITM

The goal of the PyTorchAVITM framework is to provide a intuitive and flexible implementation of the AVITM model developed by Srivastava and Sutton 2017.  This builds upon previous implementations in several key components of the inference network archtecture such as greater flexibility in the depth of the inference network, the regularization (dropout) to be used, a choice of activation function and the ability to learn the prior parameters.  We also allow robust control of the optimization procedure.  The framework provides a clean, high level API to control these decisions and easitly experiment with a larger hypthesis space of models.

# Hyper-Parameters

* input_size : Dimension of the input data
* n_components : The number of components (topics)
* item model_type : The model type, prodLDA or LDA
* hidden_sizes : Tuple of the hidden dimension for each layer in the inference network.
* activation : The activation function, softplus or relu
* dropout : The dropout rate
* learn_priors : Set priors to be learnable parameters
* batch_size : The batch size for training
* lr : The learning rate for training
* momentum : The momentum for training
* solver : The optimization method, adam or sgd
* num_epochs : The number of epochs for training
* reduce_on_plateau : Set the learning rate to reduce by a factor of 10 on a plateau of the variational objective.

# Example

![Alt text](/call_signature.png?raw=true)

The example above shows the typical usage of the PyTorch AVITM framework.  We define the input data as a PyTorch Dataset class that includes the mapping between token indexes and tokens in our vocabulary.  Next, we instantiate an AVITM model with the desired hyper-parameter settings.  Calling fit on the instantiated model will train the inference network which can subsequently be scored using the Palmetto Project scoring server.  We can also return the topics learned by the model.

# Citation

Please cite this work if you use it.

```
@MISC {Carrow2018,
    author       = "Stephen Carrow",
    title        = "PyTorchAVITM: Open Source AVITM Implementation in PyTorch",
    howpublished = "Github",
    month        = "dec",
    year         = "2018"
}
```
