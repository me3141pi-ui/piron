PiRon

PiRon is a very basic neural network framework written using JAX.
It is made for learning how forward pass, backward pass, and parameter updates work.

The framework has three main parts:

pillar → a single layer (weights, bias, activation, forward/backward).

pillarMC → manages multiple layers and connects them.

piron → training engine (optimizers, batches, epochs).

Features

Forward and backward propagation.

Stores weights, biases, and gradients.

Simple support for optimizers (SGD, Adam).

Gradient accumulation for batch training.

Reset functions for cache, gradients, and optimizers.
