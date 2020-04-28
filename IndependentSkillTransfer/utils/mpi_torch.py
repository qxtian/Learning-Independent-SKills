import torch
from utils.mpi_tools import broadcast, mpi_avg


def sync_all_params(param, root=0):
    data = torch.nn.utils.parameters_to_vector(param).detach().numpy()
    broadcast(data, root)
    torch.nn.utils.vector_to_parameters(torch.from_numpy(data), param)
    # broadcast(param, root)


def average_gradients(param_groups):
    for param_group in param_groups:
        for p in param_group['params']:
            if p.requires_grad and p.grad is not None:
                p.grad.data.copy_(torch.Tensor(mpi_avg(p.grad.data.numpy())))