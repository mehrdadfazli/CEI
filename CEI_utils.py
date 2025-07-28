def setup_injection_hook(model, injection_layer, context_embedding, alpha):
    """
    Sets up a forward hook that modifies the last input token's hidden state during generation.

    Args:
    - model: The vision-language model.
    - injection_layer: The transformer layer where the modification occurs.
    - context_embedding: The precomputed embedding to inject.
    - alpha: The weighting factor (between 0 and 1).

    Returns:
    - hook_handle: The handle to the registered hook (can be removed later).
    """

    def injection_hook(module, input, output):
        """
        Modifies the last input token's hidden state with a weighted average of itself and context_embedding.
        """
        hidden_states = output[0].clone()  # Clone to prevent in-place modification issues
        batch_size, seq_len, hidden_dim = hidden_states.shape      

        # Ensure context_embedding is the same device & dtype as hidden_states
        context_embedding_device = context_embedding.to(hidden_states.device).type(hidden_states.dtype)

        # Modify the last input token's hidden state
        hidden_states[:, -1, :] = (1 - alpha) * hidden_states[:, -1, :] + alpha * context_embedding_device

        return (hidden_states, output[1])  # Ensure tuple format matches expected model output


    # Register the hook at the specified injection layer
    hook_handle = model.language_model.model.layers[injection_layer].register_forward_hook(injection_hook)

    return hook_handle