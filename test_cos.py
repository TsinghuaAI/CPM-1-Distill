import torch
import torch.nn.functional as F

class _ParallelCosineEmbeddingLoss(torch.autograd.Function):

    '''
    NOTE: Only for target = 1
    '''
    @staticmethod
    def forward(cls, logits_x: torch.Tensor, logits_y: torch.Tensor):
        dot_prod = (logits_x * logits_y).sum(-1)

        len_x_sqrt = (logits_x * logits_x).sum(-1)
        len_x = torch.sqrt(len_x_sqrt)

        len_y_sqrt = (logits_y * logits_y).sum(-1)
        len_y = torch.sqrt(len_y_sqrt)

        len_prod = len_x * len_y

        cos_output = dot_prod / len_prod

        cls.save_for_backward(logits_x, logits_y, len_x_sqrt, len_y_sqrt, len_prod, cos_output)

        return 1 - cos_output

    @staticmethod
    def backward(cls, grad_output: torch.Tensor):
        logits_x, logits_y, len_x_sqrt, len_y_sqrt, len_prod, cos_output = cls.saved_tensors

        grad_input_x = (cos_output.unsqueeze(-1) * logits_x / len_x_sqrt.unsqueeze(-1) - logits_y / len_prod.unsqueeze(-1)) * grad_output.unsqueeze(-1)
        grad_input_y = (cos_output.unsqueeze(-1) * logits_y / len_y_sqrt.unsqueeze(-1) - logits_x / len_prod.unsqueeze(-1)) * grad_output.unsqueeze(-1)

        return grad_input_x, grad_input_y


def my_cos_loss(x, y) -> torch.Tensor:
    return _ParallelCosineEmbeddingLoss.apply(x, y)


t = torch.rand(2, 8, requires_grad=True)
a = torch.rand(2, 8, requires_grad=True)
target = t.new(t.size(0)).fill_(1)

t1 = t.clone().detach().requires_grad_(True)
a1 = a.clone().detach().requires_grad_(True)

res = F.cosine_embedding_loss(t, a, target, reduction="none")
print(res)
res = res.sum()
res.backward()
print(t.grad)
print(a.grad)

my_res = my_cos_loss(t1, a1)
print(my_res)
my_res = my_res.sum()
my_res.backward()
print(t1.grad)
print(a1.grad)
