class KollenPollackFunction(torch.autograd.Function):
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
        if input_needs_grad:
            pass

        weight_needs_grad = context.needs_input_grad[1]
        if weight_needs_grad:
            grad_weight = input.t() @ (output_for_update - (input.mm(weight.t())))
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

            grad_weight = grad_weight.t()

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
