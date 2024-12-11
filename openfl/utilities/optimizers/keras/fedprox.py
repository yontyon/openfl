# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""FedProx Keras optimizer module."""

import tensorflow as tf
import tensorflow.keras as keras


@keras.utils.register_keras_serializable()
class FedProxOptimizer(keras.optimizers.Optimizer):
    """FedProx optimizer (Keras3 based API).

    Implements the FedProx algorithm as a Keras optimizer. FedProx is a
    federated learning optimization algorithm designed to handle non-IID data.
    It introduces a proximal term to the federated averaging algorithm to
    reduce the impact of devices with outlying updates.

    Paper: https://arxiv.org/pdf/1812.06127.pdf

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        mu (float): The proximal term coefficient.
    """
    def __init__(
        self,
        learning_rate=0.01,
        mu=0.0,
        name="FedProxOptimizer",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            **kwargs,
        )
        self.mu = mu

    def build(self, variables):
        """Initialize optimizer variables.

        Args:
          var_list: list of model variables to build FedProx variables on.
        """
        if self.built:
            return
        super().build(variables)
        self.vstars = []
        for variable in variables:
            self.vstars.append(
                self.add_variable_from_reference(
                    reference_variable=variable, name="vstar"
                )
            )

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr_t = tf.cast(learning_rate, variable.dtype)
        mu_t = tf.cast(self.mu, variable.dtype)
        gradient_t = tf.cast(gradient, variable.dtype)
        vstar = self.vstars[self._get_variable_index(variable)]

        self.assign_sub(variable, lr_t * (gradient_t + mu_t * (variable - vstar)))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mu": self.mu,
            }
        )
        return config
