import torch
import torch.nn.functional as F

class _ParallelMSELoss(torch.autograd.Function):

    @staticmethod
    def forward(cls, logits: torch.Tensor, targets: torch.Tensor):
        diff = logits - targets
        square_diff = diff * diff

        cls.save_for_backward(diff)

        output = square_diff.sum(-1)

        return output

    @staticmethod
    def backward(cls, grad_output: torch.Tensor):
        diff, = cls.saved_tensors
        grad_input = 2 * diff * grad_output.unsqueeze(-1)
        return grad_input, None


def my_mse_loss(l, t) -> torch.Tensor:
    return _ParallelMSELoss.apply(l, t)


t = torch.rand(4, 2, 8, requires_grad=True)
a = torch.rand(4, 2, 8, requires_grad=False)

t1 = t.clone().detach().requires_grad_(True)
a1 = a.clone().detach().requires_grad_(False)

res = F.mse_loss(t, a, reduction="none").sum(-1)
print(res)
res = res.sum()
res.backward()
print(t.grad)

my_res = my_mse_loss(t1, a1)
print(my_res)
my_res = my_res.sum()
my_res.backward()
print(t1.grad)
