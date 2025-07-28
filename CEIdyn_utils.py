import torch
import torch.nn.functional as F

def compute_similarities(status_quota, visual_tokens):
    """
    Compute cosine similarities between the status quota embedding and visual tokens.
    
    Args:
        status_quota (torch.Tensor): Embedding representing current context [batch_size, hidden_dim].
        visual_tokens (torch.Tensor): Visual token embeddings [batch_size, num_tokens, hidden_dim].
    
    Returns:
        torch.Tensor: Similarity scores [batch_size, num_tokens].
    """
    return F.cosine_similarity(status_quota.unsqueeze(1), visual_tokens, dim=-1)

def select_context_embedding(status_quota, visual_tokens, strategy, k=3):
    """
    Select context embedding based on the specified strategy.
    
    Args:
        status_quota (torch.Tensor): Status quo embedding [batch_size, hidden_dim].
        visual_tokens (torch.Tensor): Visual token embeddings [batch_size, num_tokens, hidden_dim].
        strategy (str): Selection strategy ('top1', 'weighted_avg', 'topk_avg').
        k (int): Number of top tokens for topk_avg strategy.
    
    Returns:
        torch.Tensor: Selected context embedding [batch_size, hidden_dim].
    """
    similarities = compute_similarities(status_quota, visual_tokens)  # [batch_size, num_tokens]
    
    if strategy == "top1":
        indices = similarities.argmax(dim=-1)  # [batch_size]
        return visual_tokens[range(len(indices)), indices]  # [batch_size, hidden_dim]
    
    elif strategy == "weighted_avg":
        weights = F.softmax(similarities, dim=-1)  # [batch_size, num_tokens]
        return (visual_tokens * weights.unsqueeze(-1)).sum(dim=1)  # [batch_size, hidden_dim]
    
    elif strategy == "topk_avg":
        _, indices = similarities.topk(k, dim=-1)  # [batch_size, k]
        selected_tokens = torch.stack([visual_tokens[i, indices[i]] for i in range(visual_tokens.size(0))])  # [batch_size, k, hidden_dim]
        return selected_tokens.mean(dim=1)  # [batch_size, hidden_dim]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def inject_visual_token(hidden_state, visual_token, alpha):
    """
    Blend the hidden state with a selected visual token based on alpha.
    
    Args:
        hidden_state (torch.Tensor): Original hidden state [batch_size, hidden_dim].
        visual_token (torch.Tensor): Selected visual token [batch_size, hidden_dim].
        alpha (float): Blending factor (0 = no injection, 1 = full injection).
    
    Returns:
        torch.Tensor: Modified hidden state [batch_size, hidden_dim].
    """
    return (1 - alpha) * hidden_state + alpha * visual_token

def setup_cei_hooks(model, batch_size, hidden_dim, visual_tokens, injection_layer, device, alpha, context_strategy="top1", topk=3):
    """
    Configure forward hooks for dynamic CEI in the language model.
    
    Args:
        model: InstructBLIP model instance.
        batch_size (int): Number of samples (typically 1 for single-image processing).
        hidden_dim (int): Hidden dimension of the language model.
        visual_tokens (torch.Tensor): Visual embeddings [batch_size, num_tokens, hidden_dim].
        injection_layer (int): Layer index for injecting visual tokens.
        device: Device for tensor operations (e.g., 'cuda').
        alpha (float): Weight for visual token injection.
        context_strategy (str): Strategy for selecting context embedding ('top1', 'weighted_avg', 'topk_avg').
        topk (int): Number of top tokens for topk_avg strategy.
    
    Returns:
        tuple: Hooks for the last layer and injection layer.
    """
    # Initialize status quota as the mean of visual tokens
    status_quota = visual_tokens.mean(dim=1).to(device).clone()
    
    def last_layer_hook(module, input, output):
        """Update status quota with the last token’s hidden state from the final layer."""
        nonlocal status_quota
        hidden_states = output[0]  # [batch_size, seq_len, hidden_dim]
        status_quota = hidden_states[:, -1, :].detach().clone()
    
    def injection_hook(module, input, output):
        """Inject a visual token into the hidden states at the specified layer."""
        nonlocal status_quota
        hidden_states = output[0].clone()  # [batch_size, seq_len, hidden_dim]
        seq_len = hidden_states.size(1)
        if seq_len >= 1:
            selected_visual_tokens = select_context_embedding(status_quota, visual_tokens, context_strategy, topk)
            hidden_states[:, -1, :] = inject_visual_token(hidden_states[:, -1, :], selected_visual_tokens, alpha)
        return (hidden_states, *output[1:])  # Preserve output tuple structure
    
    # Register hooks on the specified layers
    hook_last = model.language_model.model.layers[-1].register_forward_hook(last_layer_hook)
    hook_injection = model.language_model.model.layers[injection_layer].register_forward_hook(injection_hook)
    
    return hook_last, hook_injection


# import torch
# import torch.nn.functional as F

# def compute_similarities(status_quota, visual_tokens):
#     """
#     Compute cosine similarities between the status quota embedding and visual tokens.
    
#     Args:
#         status_quota (torch.Tensor): Embedding representing current context [batch_size, hidden_dim].
#         visual_tokens (torch.Tensor): Visual token embeddings [batch_size, num_tokens, hidden_dim].
    
#     Returns:
#         torch.Tensor: Similarity scores [batch_size, num_tokens].
#     """
#     return F.cosine_similarity(status_quota.unsqueeze(1), visual_tokens, dim=-1)

# def inject_visual_token(hidden_state, visual_token, alpha):
#     """
#     Blend the hidden state with a selected visual token based on alpha.
    
#     Args:
#         hidden_state (torch.Tensor): Original hidden state [batch_size, hidden_dim].
#         visual_token (torch.Tensor): Selected visual token [batch_size, hidden_dim].
#         alpha (float): Blending factor (0 = no injection, 1 = full injection).
    
#     Returns:
#         torch.Tensor: Modified hidden state [batch_size, hidden_dim].
#     """
#     return (1 - alpha) * hidden_state + alpha * visual_token

# def setup_cei_hooks(model, batch_size, hidden_dim, visual_tokens, injection_layer, device, alpha):
#     """
#     Configure forward hooks for dynamic CEI in the language model.
    
#     Args:
#         model: InstructBLIP model instance.
#         batch_size (int): Number of samples (typically 1 for single-image processing).
#         hidden_dim (int): Hidden dimension of the language model.
#         visual_tokens (torch.Tensor): Visual embeddings [batch_size, num_tokens, hidden_dim].
#         injection_layer (int): Layer index for injecting visual tokens.
#         device: Device for tensor operations (e.g., 'cuda').
#         alpha (float): Weight for visual token injection.
    
#     Returns:
#         tuple: Hooks for the last layer and injection layer.
#     """
#     # Initialize status quota as the mean of visual tokens
#     status_quota = visual_tokens.mean(dim=1).to(device).clone()
    
#     def last_layer_hook(module, input, output):
#         """Update status quota with the last token’s hidden state from the final layer."""
#         nonlocal status_quota
#         hidden_states = output[0]  # [batch_size, seq_len, hidden_dim]
#         status_quota = hidden_states[:, -1, :].detach().clone()
    
#     def injection_hook(module, input, output):
#         """Inject a visual token into the hidden states at the specified layer."""
#         nonlocal status_quota
#         hidden_states = output[0].clone()  # [batch_size, seq_len, hidden_dim]
#         seq_len = hidden_states.size(1)
#         if seq_len >= 1:
#             similarities = compute_similarities(status_quota, visual_tokens)
#             indices = similarities.argmax(dim=-1)  # [batch_size]
#             selected_visual_tokens = visual_tokens[range(batch_size), indices]  # [batch_size, hidden_dim]
#             hidden_states[:, -1, :] = inject_visual_token(hidden_states[:, -1, :], selected_visual_tokens, alpha)
#         return (hidden_states, *output[1:])  # Preserve output tuple structure
    
#     # Register hooks on the specified layers
#     hook_last = model.language_model.model.layers[-1].register_forward_hook(last_layer_hook)
#     hook_injection = model.language_model.model.layers[injection_layer].register_forward_hook(injection_hook)
    
#     return hook_last, hook_injection