import numpy as np
import torch
import pdb


def sigmoid(X):
    """
    Returns the sigmoid function, i.e. 1/(1+exp(-X))
    """

    # to avoid runtime warnings, if abs(X) is more than 500, we just cap it there
    Y = (
        X.copy()
    )  # this ensures we don't overwrite entries in X - Python can be a trickster!
    toobig = X > 500
    toosmall = X < -500
    Y[toobig] = 500
    Y[toosmall] = -500

    return 1.0 / (1.0 + np.exp(-Y))


def ReLU(X):
    """
    Returns the ReLU function, i.e. X if X > 0, 0 otherwise
    """

    # to avoid runtime warnings, if abs(X) is more than 500, we just cap it there
    Y = (
        X.copy()
    )  # this ensures we don't overwrite entries in X - Python can be a trickster!
    neg = X < 0
    Y[neg] = 0

    return Y


def add_bias(inputs):
    """
    Append an "always on" bias unit to some inputs
    """
    print(inputs.shape)
    return np.append(inputs, np.ones((1, inputs.shape[1])), axis=0)


# Creates a random set of batches, returns an array of indices, one for each batch
def create_batches(rng, batch_size, num_samples):
    """
    For a given number of samples, returns an array of indices of random batches of the specified size.

    If the size of the data is not divisible by the batch size some samples will not be included.
    """

    # determine the total number of batches
    num_batches = int(np.floor(num_samples / batch_size))

    # get the batches (without replacement)
    return rng.choice(
        np.arange(num_samples), size=(num_batches, batch_size), replace=False
    )


class MultiLayerPerceptron(torch.nn.Module):
    """
    Simple multilayer perceptron model class with one hidden layer.
    """

    def __init__(
        self,
        num_inputs=784,
        num_hidden=100,
        num_outputs=10,
        activation_type="sigmoid",
        bias=False,
    ):
        """
        Initializes a multilayer perceptron with a single hidden layer.

        Arguments:
        - num_inputs (int, optional): number of input units (i.e., image size)
        - num_hidden (int, optional): number of hidden units in the hidden layer
        - num_outputs (int, optional): number of output units (i.e., number of
          classes)
        - activation_type (str, optional): type of activation to use for the hidden
          layer ('sigmoid', 'tanh', 'relu' or 'linear')
        - bias (bool, optional): if True, each linear layer will have biases in
          addition to weights
        """

        super().__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.activation_type = activation_type
        self.bias = bias  # boolean

        # default weights (and biases, if applicable) initialization is used
        # see https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
        self.lin1 = torch.nn.Linear(num_inputs, num_hidden, bias=bias)
        self.lin2 = torch.nn.Linear(num_hidden, num_outputs, bias=bias)

        self._store_initial_weights_biases()

        self._set_activation()  # activation on the hidden layer
        self.softmax = torch.nn.Softmax(dim=1)  # activation on the output layer

    def _store_initial_weights_biases(self):
        """
        Stores a copy of the network's initial weights and biases.
        """

        self.init_lin1_weight = self.lin1.weight.data.clone()
        self.init_lin2_weight = self.lin2.weight.data.clone()
        if self.bias:
            self.init_lin1_bias = self.lin1.bias.data.clone()
            self.init_lin2_bias = self.lin2.bias.data.clone()

    def _set_activation(self):
        """
        Sets the activation function used for the hidden layer.
        """

        if self.activation_type.lower() == "sigmoid":
            self.activation = torch.nn.Sigmoid()  # maps to [0, 1]
        elif self.activation_type.lower() == "tanh":
            self.activation = torch.nn.Tanh()  # maps to [-1, 1]
        elif self.activation_type.lower() == "relu":
            self.activation = torch.nn.ReLU()  # maps to positive
        elif self.activation_type.lower() == "identity":
            self.activation = torch.nn.Identity()  # maps to same
        else:
            raise NotImplementedError(
                f"{self.activation_type} activation type not recognized. Only "
                "'sigmoid', 'relu' and 'identity' have been implemented so far."
            )

    def forward(self, X, y=None):
        """
        Runs a forward pass through the network.

        Arguments:
        - X (torch.Tensor): Batch of input images.
        - y (torch.Tensor, optional): Batch of targets. This variable is not used
          here. However, it may be needed for other learning rules, to it is
          included as an argument here for compatibility.

        Returns:
        - y_pred (torch.Tensor): Predicted targets.
        """

        l1_input = self.lin1(X.reshape(-1, self.num_inputs))
        h = self.activation(l1_input)
        l2_input = self.lin2(h)
        y_pred = self.softmax(l2_input)
        return y_pred

    def forward_backprop(self, X):
        """
        Identical to forward(). Should not be overwritten when creating new
        child classes to implement other learning rules, as this method is used
        to compare the gradients calculated with other learning rules to those
        calculated with backprop.
        """

        h = self.activation(self.lin1(X.reshape(-1, self.num_inputs)))
        y_pred = self.softmax(self.lin2(h))
        return y_pred

    def list_parameters(self):
        """
        Returns a list of model names for a gradient dictionary.

        Returns:
        - params_list (list): List of parameter names.
        """

        params_list = list()

        for layer_str in ["lin1", "lin2"]:
            params_list.append(f"{layer_str}_weight")
            if self.bias:
                params_list.append(f"{layer_str}_bias")

        return params_list

    def gather_gradient_dict(self):
        """
        Gathers a gradient dictionary for the model's parameters. Raises a
        runtime error if any parameters have no gradients.

        Returns:
        - gradient_dict (dict): A dictionary of gradients for each parameter.
        """

        params_list = self.list_parameters()

        gradient_dict = dict()
        for param_name in params_list:
            layer_str, param_str = param_name.split("_")  # lin1_parameterName
            layer = getattr(self, layer_str)  # self.lin1
            grad = getattr(layer, param_str).grad  # slef.lin1.grad
            if grad is None:
                raise RuntimeError("No gradient was computed")
            gradient_dict[param_name] = grad.detach().clone().numpy()

        return gradient_dict


# Calculate the accuracy of the network on some data
def calculate_accuracy(outputs, targets):
    """
    Calculate the accuracy in categorization of some outputs given some targets.
    """

    # binarize the outputs for an easy calculation
    categories = (outputs == np.tile(outputs.max(axis=0), (10, 1))).astype("float")

    # get the accuracy
    accuracy = np.sum(categories * targets) / targets.shape[1]

    return accuracy * 100.0


def calculate_cosine_similarity(grad_1, grad_2):
    """
    Calculate the cosine similarity between two gradients
    """
    grad_1 = grad_1.flatten()
    grad_2 = grad_2.flatten()
    return (
        np.dot(grad_1, grad_2)
        / np.sqrt(np.dot(grad_1, grad_1))
        / np.sqrt(np.dot(grad_2, grad_2))
    )


def calculate_grad_snr(grad, epsilon=1e-3):
    """
    Calculate the average SNR |mean|/std across all parameters in a gradient update
    """
    return np.mean(np.abs(np.mean(grad, axis=0)) / (np.std(grad, axis=0) + epsilon))


class MLP(object):
    """
    The class for creating and training a two-layer perceptron.
    """

    # The initialization function
    def __init__(self, rng, N=100, sigma=1.0, activation="sigmoid"):
        """
        The initialization function for the MLP.

         - N is the number of hidden units
         - sigma is the SD for initializing the weights
         - activation is the function to use for unit activity, options are 'sigmoid' and 'ReLU'
        """

        # store the variables for easy access
        self.N = N
        self.sigma = sigma
        self.activation = activation

        # initialize the weights
        self.W_h = rng.normal(
            scale=self.sigma, size=(self.N, 784 + 1)
        )  # input-to-hidden weights & bias
        self.W_y = rng.normal(
            scale=self.sigma, size=(10, self.N + 1)
        )  # hidden-to-output weights & bias
        self.B = rng.normal(scale=self.sigma, size=(self.N, 10))  # feedback weights

    # The non-linear activation function
    def activate(self, inputs):
        """
        Pass some inputs through the activation function.
        """
        if self.activation == "sigmoid":
            Y = sigmoid(inputs)
        elif self.activation == "ReLU":
            Y = ReLU(inputs)
        else:
            raise Exception("Unknown activation function")
        return Y

    # The function for performing a forward pass up through the network during inference
    def inference(self, rng, inputs, W_h=None, W_y=None, noise=0.0):
        """
        Recognize inputs, i.e. do a forward pass up through the network. If desired, alternative weights
        can be provided
        """

        # load the current network weights if no weights given
        if W_h is None:
            W_h = self.W_h
        if W_y is None:
            W_y = self.W_y

        # calculate the hidden activities
        hidden = self.activate(np.dot(W_h, add_bias(inputs)))
        if not (noise == 0.0):
            hidden += rng.normal(scale=noise, size=hidden.shape)

        # calculate the output activities
        output = self.activate(np.dot(W_y, add_bias(hidden)))

        if not (noise == 0.0):
            output += rng.normal(scale=noise, size=output.shape)

        return hidden, output

    # A function for calculating the derivative of the activation function
    def act_deriv(self, activity):
        """
        Calculate the derivative of some activations with respect to the inputs
        """
        if self.activation == "sigmoid":
            derivative = activity * (1 - activity)
        elif self.activation == "ReLU":
            derivative = 1.0 * (activity > 1)
        else:
            raise Exception("Unknown activation function")
        return derivative

    def mse_loss_batch(self, rng, inputs, targets, W_h=None, W_y=None, output=None):
        """
        Calculate the mean-squared error loss on the given targets (average over the batch)
        """

        # do a forward sweep through the network
        if output is None:
            (hidden, output) = self.inference(rng, inputs, W_h, W_y)
        return np.sum((targets - output) ** 2, axis=0)

    # The function for calculating the mean-squared error loss
    def mse_loss(self, rng, inputs, targets, W_h=None, W_y=None, output=None):
        """
        Calculate the mean-squared error loss on the given targets (average over the batch)
        """
        return np.mean(
            self.mse_loss_batch(rng, inputs, targets, W_h=W_h, W_y=W_y, output=output)
        )

    # function for calculating perturbation updates
    def perturb(self, rng, inputs, targets, noise=1.0):
        """
        Calculates the weight updates for perturbation learning, using noise with SD as given
        """

        # get the random perturbations
        delta_W_h = rng.normal(scale=noise, size=self.W_h.shape)
        delta_W_y = rng.normal(scale=noise, size=self.W_y.shape)

        # calculate the loss with and without the perturbations
        loss_now = self.mse_loss(rng, inputs, targets)
        loss_per = self.mse_loss(
            rng, inputs, targets, self.W_h + delta_W_h, self.W_y + delta_W_y
        )

        # updates
        delta_loss = loss_now - loss_per
        W_h_update = delta_loss * delta_W_h / noise**2
        W_y_update = delta_loss * delta_W_y / noise**2
        return W_h_update, W_y_update

    def node_perturb(self, rng, inputs, targets, noise=1.0):
        """
        Calculates the weight updates for node perturbation learning, using noise with SD as given
        """

        # get the random perturbations
        hidden, output = self.inference(rng, inputs)
        hidden_p, output_p = self.inference(rng, inputs, noise=noise)

        loss_now = self.mse_loss_batch(rng, inputs, targets, output=output)
        loss_per = self.mse_loss_batch(rng, inputs, targets, output=output_p)
        delta_loss = loss_now - loss_per

        hidden_update = np.mean(
            delta_loss
            * (
                ((hidden_p - hidden) / noise**2)[:, None, :]
                * add_bias(inputs)[None, :, :]
            ),
            axis=2,
        )
        output_update = np.mean(
            delta_loss
            * (
                ((output_p - output) / noise**2)[:, None, :]
                * add_bias(hidden_p)[None, :, :]
            ),
            axis=2,
        )

        return (hidden_update, output_update)

    # function for calculating gradient updates
    def gradient(self, rng, inputs, targets):
        """
        Calculates the weight updates for gradient descent learning
        """

        # do a forward pass
        hidden, output = self.inference(rng, inputs)

        # calculate the gradients
        error = targets - output
        delta_W_h = np.dot(
            np.dot(self.W_y[:, :-1].transpose(), error * self.act_deriv(output))
            * self.act_deriv(hidden),
            add_bias(inputs).transpose(),
        )
        delta_W_y = np.dot(error * self.act_deriv(output), add_bias(hidden).transpose())

        return delta_W_h, delta_W_y

    # function for calculating feedback alignment updates
    def feedback(self, rng, inputs, targets):
        """
        Calculates the weight updates for feedback alignment learning
        """

        # do a forward pass
        hidden, output = self.inference(rng, inputs)

        # calculate the updates
        error = targets - output
        delta_W_h = np.dot(
            np.dot(self.B, error * self.act_deriv(output)) * self.act_deriv(hidden),
            add_bias(inputs).transpose(),
        )
        delta_W_y = np.dot(error * self.act_deriv(output), add_bias(hidden).transpose())

        return delta_W_h, delta_W_y

    # function for calculating Kolen-Pollack updates
    def kolepoll(self, rng, inputs, targets, eta_back=0.01):
        """
        Calculates the weight updates for Kolen-Polack learning
        """

        # do a forward pass
        (hidden, output) = self.inference(rng, inputs)

        # calculate the updates for the forward weights
        error = targets - output
        delta_W_h = np.dot(
            np.dot(self.B, error * self.act_deriv(output)) * self.act_deriv(hidden),
            add_bias(inputs).transpose(),
        )
        delta_err = np.dot(error * self.act_deriv(output), add_bias(hidden).transpose())
        delta_W_y = delta_err - 0.1 * self.W_y

        # calculate the updates for the backwards weights and implement them
        delta_B = delta_err[:, :-1].transpose() - 0.1 * self.B
        self.B += eta_back * delta_B
        return (delta_W_h, delta_W_y)

    def return_grad(
        self, rng, inputs, targets, algorithm="backprop", eta=0.0, noise=1.0
    ):
        # calculate the updates for the weights with the appropriate algorithm
        if algorithm == "perturb":
            delta_W_h, delta_W_y = self.perturb(rng, inputs, targets, noise=noise)
        elif algorithm == "node_perturb":
            delta_W_h, delta_W_y = self.node_perturb(rng, inputs, targets, noise=noise)
        elif algorithm == "feedback":
            delta_W_h, delta_W_y = self.feedback(rng, inputs, targets)
        elif algorithm == "kolepoll":
            delta_W_h, delta_W_y = self.kolepoll(rng, inputs, targets, eta_back=eta)
        else:
            delta_W_h, delta_W_y = self.gradient(rng, inputs, targets)

        return delta_W_h, delta_W_y

    # function for updating the network
    def update(self, rng, inputs, targets, algorithm="backprop", eta=0.01, noise=1.0):
        """
        Updates the synaptic weights (and unit biases) using the given algorithm, options are:

        - 'backprop': backpropagation-of-error (default)
        - 'perturb' : weight perturbation (use noise with SD as given)
        - 'feedback': feedback alignment
        - 'kolepoll': Kolen-Pollack
        """

        delta_W_h, delta_W_y = self.return_grad(
            rng, inputs, targets, algorithm=algorithm, eta=eta, noise=noise
        )

        # do the updates
        self.W_h += eta * delta_W_h
        self.W_y += eta * delta_W_y

    # train the network using the update functions
    def train(
        self,
        rng,
        images,
        labels,
        num_epochs,
        test_images,
        test_labels,
        learning_rate=0.01,
        batch_size=20,
        algorithm="backprop",
        noise=1.0,
        report=False,
        report_rate=10,
    ):
        """
        Trains the network with algorithm in batches for the given number of epochs on the data provided.

        Uses batches with size as indicated by batch_size and given learning rate.

        For perturbation methods, uses SD of noise as given.

        Categorization accuracy on a test set is also calculated.

        Prints a message every report_rate epochs if requested.

        Returns an array of the losses achieved at each epoch (and accuracies if test data given).
        """

        # provide an output message
        if report:
            print("Training starting...")

        # make batches from the data
        batches = create_batches(rng, batch_size, images.shape[1])

        # create arrays to store loss and accuracy values
        losses = np.zeros((num_epochs * batches.shape[0],))
        accuracy = np.zeros((num_epochs,))
        cosine_similarity = np.zeros((num_epochs,))

        # estimate the gradient SNR on the test set
        grad = np.zeros((test_images.shape[1], *self.W_h.shape))
        for t in range(test_images.shape[1]):
            inputs = test_images[:, [t]]
            targets = test_labels[:, [t]]
            grad[t, ...], _ = self.return_grad(
                rng, inputs, targets, algorithm=algorithm, eta=0.0, noise=noise
            )
        snr = calculate_grad_snr(grad)
        # run the training for the given number of epochs
        update_counter = 0
        for epoch in range(num_epochs):

            # step through each batch
            for b in range(batches.shape[0]):
                # get the inputs and targets for this batch
                inputs = images[:, batches[b, :]]
                targets = labels[:, batches[b, :]]

                # calculate the current loss
                losses[update_counter] = self.mse_loss(rng, inputs, targets)

                # update the weights
                self.update(
                    rng,
                    inputs,
                    targets,
                    eta=learning_rate,
                    algorithm=algorithm,
                    noise=noise,
                )
                update_counter += 1

            # calculate the current test accuracy
            (testhid, testout) = self.inference(rng, test_images)
            accuracy[epoch] = calculate_accuracy(testout, test_labels)
            grad_test, _ = self.return_grad(
                rng, test_images, test_labels, algorithm=algorithm, eta=0.0, noise=noise
            )
            grad_bp, _ = self.return_grad(
                rng,
                test_images,
                test_labels,
                algorithm="backprop",
                eta=0.0,
                noise=noise,
            )
            cosine_similarity[epoch] = calculate_cosine_similarity(grad_test, grad_bp)

            # print an output message every 10 epochs
            if report and np.mod(epoch + 1, report_rate) == 0:
                print(
                    "...completed ",
                    epoch + 1,
                    " epochs of training. Current loss: ",
                    round(losses[update_counter - 1], 2),
                    ".",
                )

        # provide an output message
        if report:
            print("Training complete.")

        return (losses, accuracy, cosine_similarity, snr)


class HebbianFunction(torch.autograd.Function):
    """
    Gradient computing function class for Hebbian learning.
    """

    @staticmethod
    def forward(
        context, input: torch.Tensor, weight, bias=None, nonlinearity=None, target=None
    ):
        """
        Forward pass method for the layer. Computes the output of the layer and
        stores variables needed for the backward pass.

        Arguments:
        - context (torch context): context in which variables can be stored for
          the backward pass.
        - input (torch tensor): input to the layer.
        - weight (torch tensor): layer weights.
        - bias (torch tensor, optional): layer biases.
        - nonlinearity (torch functional, optional): nonlinearity for the layer.
        - target (torch tensor, optional): layer target, if applicable.

        Returns:
        - output (torch tensor): layer output.
        """

        # compute the output for the layer (linear layer with non-linearity)
        output = input.mm(weight.t())  # Matrix multiplication
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        if nonlinearity is not None:
            output = nonlinearity(output)

        # calculate the output to use for the backward pass
        output_for_update = (
            output if target is None else target
        )  # Train on target or not? if not use the actual output

        # store variables in the context for the backward pass
        context.save_for_backward(input, weight, bias, output_for_update)

        return output

    @staticmethod
    def backward(context, grad_output=None):
        """
        Backward pass method for the layer. Computes and returns the gradients for
        all variables passed to forward (returning None if not applicable).

        Arguments:
        - context (torch context): context in which variables can be stored for
          the backward pass.
        - input (torch tensor): input to the layer.
        - weight (torch tensor): layer weights.
        - bias (torch tensor, optional): layer biases.
        - nonlinearity (torch functional, optional): nonlinearity for the layer.
        - target (torch tensor, optional): layer target, if applicable.

        Returns:
        - grad_input (None): gradients for the input (None, since gradients are not
          backpropagated in Hebbian learning).
        - grad_weight (torch tensor): gradients for the weights.
        - grad_bias (torch tensor or None): gradients for the biases, if they aren't
          None.
        - grad_nonlinearity (None): gradients for the nonlinearity (None, since
          gradients do not apply to the non-linearities).
        - grad_target (None): gradients for the targets (None, since
          gradients do not apply to the targets).
        """

        input, weight, bias, output_for_update = context.saved_tensors
        grad_input = None
        grad_weight = None
        grad_bias = None
        grad_nonlinearity = None
        grad_target = None

        input_needs_grad = context.needs_input_grad[0]
        if input_needs_grad:  # What is pass mean in this case? Why pass?
            pass

        weight_needs_grad = context.needs_input_grad[1]
        if weight_needs_grad:
            grad_weight = output_for_update.t().mm(input)
            grad_weight = grad_weight / len(input)  # average across batch

            # center around 0
            grad_weight = grad_weight - grad_weight.mean(
                axis=0
            )  # center around 0 -> ? Suspect to be one of normalization method

            # or apply Oja's rule (not compatible with clamping outputs to the targets!)
            #  oja_subtract = output_for_update.pow(2).mm(grad_weight).mean(axis=0)
            #  grad_weight = grad_weight - oja_subtract

            # take the negative, as the gradient will be subtracted
            grad_weight = -grad_weight

        if bias is not None:
            bias_needs_grad = context.needs_input_grad[2]
            if bias_needs_grad:
                grad_bias = output_for_update.mean(axis=0)  # average across batch

                # center around 0
                grad_bias = grad_bias - grad_bias.mean()

                # or apply an adaptation of Oja's rule for biases
                # (not compatible with clamping outputs to the targets!)
                # oja_subtract = (output_for_update.pow(2) * bias).mean(axis=0)
                # grad_bias = grad_bias - oja_subtract

                # take the negative, as the gradient will be subtracted
                grad_bias = -grad_bias

        return grad_input, grad_weight, grad_bias, grad_nonlinearity, grad_target


class HebbianMultiLayerPerceptron(MultiLayerPerceptron):
    """
    Hebbian multilayer perceptron with one hidden layer.
    """

    def __init__(self, clamp_output=True, **kwargs):
        """
        Initializes a Hebbian multilayer perceptron object

        Arguments:
        - clamp_output (bool, optional): if True, outputs are clamped to targets,
          if available, when computing weight updates.
        """

        self.clamp_output = clamp_output
        super().__init__(**kwargs)

    def forward(self, X, y=None):
        """
        Runs a forward pass through the network.

        Arguments:
        - X (torch.Tensor): Batch of input images.
        - y (torch.Tensor, optional): Batch of targets, stored for the backward
          pass to compute the gradients for the last layer.

        Returns:
        - y_pred (torch.Tensor): Predicted targets.
        """

        h = HebbianFunction.apply(
            X.reshape(-1, self.num_inputs),
            self.lin1.weight,
            self.lin1.bias,
            self.activation,
        )

        # if targets are provided, they can be used instead of the last layer's
        # output to train the last layer.
        if y is None or not self.clamp_output:
            targets = None
        else:
            targets = torch.nn.functional.one_hot(
                y, num_classes=self.num_outputs
            ).float()

        y_pred = HebbianFunction.apply(
            h, self.lin2.weight, self.lin2.bias, self.softmax, targets
        )

        return y_pred


class HebbianBackpropMultiLayerPerceptron(MultiLayerPerceptron):
    """
    Hybrid backprop/Hebbian multilayer perceptron with one hidden layer.
    """

    def forward(self, X, y=None):
        """
        Runs a forward pass through the network.

        Arguments:
        - X (torch.Tensor): Batch of input images.
        - y (torch.Tensor, optional): Batch of targets, not used here.

        Returns:
        - y_pred (torch.Tensor): Predicted targets.
        """

        # Hebbian layer
        h = HebbianFunction.apply(
            X.reshape(-1, self.num_inputs),
            self.lin1.weight,
            self.lin1.bias,
            self.activation,
        )

        # backprop layer
        y_pred = self.softmax(self.lin2(h))

        return y_pred


class WeightPerturbMLP(MLP):
    """
    A multilayer perceptron that is capable of learning through weight perturbation
    """

    def perturb(self, rng, inputs, targets, noise=1.0):
        """
        Calculates the weight updates for perturbation learning, using noise with SD as given
        """

        # get the random perturbations
        delta_W_h = rng.normal(scale=noise, size=self.W_h.shape)
        delta_W_y = rng.normal(scale=noise, size=self.W_y.shape)

        # calculate the loss with and without the perturbations
        loss_now = self.mse_loss(rng, inputs, targets)
        loss_per = self.mse_loss(
            rng, inputs, targets, self.W_h + delta_W_h, self.W_y + delta_W_y
        )

        # updates
        delta_loss = loss_now - loss_per
        W_h_update = delta_loss * delta_W_h / noise**2
        W_y_update = delta_loss * delta_W_y / noise**2
        return W_h_update, W_y_update

    def perturb_loss(self, rng, inputs, targets):
        """
        Calculates the perturbation loss for Weight Perturbation MLP.
        """
        delta_W_h, delta_W_y = self.perturb(rng, inputs, targets)
        perturb_loss = self.mse_loss(
            rng, inputs, targets, W_h=self.W_h + delta_W_h, W_y=self.W_y + delta_W_y
        )
        return perturb_loss


class NodePerturbMLP(MLP):
    """
    A multilayer perceptron that is capable of learning through node perturbation
    """

    def node_perturb(self, rng, inputs, targets, noise=1.0):
        """
        Calculates the weight updates for node perturbation learning, using noise with SD as given
        """

        # get the random perturbations
        hidden, output = self.inference(rng, inputs)
        hidden_p, output_p = self.inference(rng, inputs, noise=noise)

        loss_now = self.mse_loss_batch(rng, inputs, targets, output=output)
        loss_per = self.mse_loss_batch(rng, inputs, targets, output=output_p)
        delta_loss = loss_now - loss_per

        hidden_update = np.mean(
            delta_loss
            * (
                ((hidden_p - hidden) / noise**2)[:, None, :]
                * add_bias(inputs)[None, :, :]
            ),
            axis=2,
        )
        output_update = np.mean(
            delta_loss
            * (
                ((output_p - output) / noise**2)[:, None, :]
                * add_bias(hidden_p)[None, :, :]
            ),
            axis=2,
        )

        return (hidden_update, output_update)

    def node_perturb_loss(self, rng, inputs, targets, noise=1.0):
        """
        Calculates the node perturbation loss for Node Perturbation MLP.
        """
        delta_W_h, delta_W_y = self.node_perturb(rng, inputs, targets, noise=noise)
        node_perturb_loss = self.mse_loss(
            rng, inputs, targets, W_h=self.W_h + delta_W_h, W_y=self.W_y + delta_W_y
        )
        return node_perturb_loss


class FeedbackAlignmentMLP(MLP):
    """
    A multilayer perceptron that is capable of learning through the Feedback Alignment algorithm
    """

    # function for calculating feedback alignment updates
    def feedback(self, rng, inputs, targets):
        """
        Calculates the weight updates for feedback alignment learning
        """

        # do a forward pass
        hidden, output = self.inference(rng, inputs)

        # calculate the updates
        error = targets - output
        delta_W_h = np.dot(
            np.dot(self.B, error * self.act_deriv(output)) * self.act_deriv(hidden),
            add_bias(inputs).transpose(),
        )
        delta_W_y = np.dot(error * self.act_deriv(output), add_bias(hidden).transpose())

        return delta_W_h, delta_W_y

    def feedback_loss(self, rng, inputs, targets):
        """
        Calculates the feedback alignment loss for Feedback Alignment MLP.
        """
        delta_W_h, delta_W_y = self.feedback(rng, inputs, targets)
        feedback_loss = self.mse_loss(
            rng, inputs, targets, W_h=self.W_h + delta_W_h, W_y=self.W_y + delta_W_y
        )
        return feedback_loss


class KolenPollackMLP(MLP):
    """
    A multilayer perceptron that is capable of learning through the Kolen-Pollack algorithm
    """

    def kolepoll(self, rng, inputs, targets, eta_back=0.01):
        """
        Calculates the weight updates for Kolen-Polack learning
        """

        # do a forward pass
        (hidden, output) = self.inference(rng, inputs)

        # calculate the updates for the forward weights
        error = targets - output
        delta_W_h = np.dot(
            np.dot(self.B, error * self.act_deriv(output)) * self.act_deriv(hidden),
            add_bias(inputs).transpose(),
        )
        delta_err = np.dot(error * self.act_deriv(output), add_bias(hidden).transpose())
        delta_W_y = delta_err - 0.1 * self.W_y

        # calculate the updates for the backwards weights and implement them
        delta_B = delta_err[:, :-1].transpose() - 0.1 * self.B
        self.B += eta_back * delta_B
        return (delta_W_h, delta_W_y)

    def kolepoll_loss(self, rng, inputs, targets, eta_back=0.01):
        """
        Calculates the Kolen-Pollack loss for Kolen-Pollack MLP.
        """
        delta_W_h, delta_W_y = self.kolepoll(rng, inputs, targets, eta_back=eta_back)
        kolepoll_loss = self.mse_loss(
            rng, inputs, targets, W_h=self.W_h + delta_W_h, W_y=self.W_y + delta_W_y
        )
        return kolepoll_loss


if __name__ == "__main__":
    pass
