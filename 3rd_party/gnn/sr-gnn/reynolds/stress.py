def compute_reynolds_stress(data):
    """
    Placeholder: returns identity tensor at each fine node.
    """
    N = data.y.shape[0]
    I = torch.eye(3, device=data.y.device).unsqueeze(0).repeat(N,1,1)
    return I