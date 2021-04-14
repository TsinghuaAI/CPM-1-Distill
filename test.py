import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(233)


class _ParallelSoftCrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(cls, logits: torch.Tensor, targets: torch.Tensor):
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(logits, dim=-1)[0]
        # torch.distributed.all_reduce(logits_max,
        #                              op=torch.distributed.ReduceOp.MAX,
        #                              group=get_model_parallel_group())
        # Subtract the maximum value.
        logits.sub_(logits_max.unsqueeze(dim=-1))
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1)
        # torch.distributed.all_reduce(sum_exp_logits,
        #                              op=torch.distributed.ReduceOp.SUM,
        #                              group=get_model_parallel_group())

        targets_max = torch.max(targets, dim=-1)[0]
        # torch.distributed.all_reduce(targets_max,
        #                              op=torch.distributed.ReduceOp.MAX,
        #                              group=get_model_parallel_group())
        # Subtract the maximum value.
        targets.sub_(targets_max.unsqueeze(dim=-1))
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_targets = targets.exp()
        sum_exp_targets = exp_targets.sum(dim=-1)
        # torch.distributed.all_reduce(sum_exp_targets,
        #                              op=torch.distributed.ReduceOp.SUM,
        #                              group=get_model_parallel_group())

        # targets_softmax: [b, s, v_p]
        targets_softmax = torch.div(exp_targets, sum_exp_targets.unsqueeze(-1))

        # sum_targets_softmax_logits: [b, s]
        sum_targets_softmax_logits = torch.matmul(
            targets_softmax.unsqueeze(-2), logits.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        # torch.distributed.all_reduce(sum_targets_softmax_logits,
        #                              op=torch.distributed.ReduceOp.SUM,
        #                              group=get_model_parallel_group())

        log_targets_softmax = torch.log(targets_softmax)
        sum_log_targets_softmax = torch.matmul(
            targets_softmax.unsqueeze(-2), log_targets_softmax.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        loss = torch.log(sum_exp_logits) - sum_targets_softmax_logits + sum_log_targets_softmax

        logits_softmax = torch.div(exp_logits, sum_exp_logits.unsqueeze(-1))

        cls.save_for_backward(logits_softmax, targets_softmax)

        return loss

    @staticmethod
    def backward(cls, grad_output: torch.Tensor):
        logits_softmax, targets_softmax = cls.saved_tensors
        grad_input = (logits_softmax - targets_softmax) * grad_output.unsqueeze(-1)

        return grad_input, None



def my_soft_ce_loss(l, t) -> torch.Tensor:
    return _ParallelSoftCrossEntropyLoss.apply(l, t)

t = torch.rand(4, 2, 8, requires_grad=True)
a = torch.rand(4, 2, 8, requires_grad=False)

t1 = t.clone().detach().requires_grad_(True)
a1 = a.clone().detach().requires_grad_(False)

res = F.kl_div(F.log_softmax(t, dim=-1), F.softmax(a, dim=-1), reduction="none").sum(-1)
print(res)
res = res.mean()
res.backward()
print(t.grad)
print(t.grad.sum(dim=0))

my_res = my_soft_ce_loss(t1, a1)
print(my_res)
my_res = my_res.mean()
my_res.backward()
print(t1.grad)
print(t1.grad.sum(dim=0))

