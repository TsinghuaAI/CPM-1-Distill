from numpy.core.fromnumeric import squeeze
import torch
import torch.nn as nn

from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size

from .utils import LogitUtility

class _ParallelSoftmax(torch.autograd.Function):
    
    @staticmethod
    def forward(cls, x_in: torch.Tensor) -> torch.Tensor:
        # Copy so the input remains unchanged.
        x = x_in.clone()
        # Maximum value along vocab dimension across all GPUs.
        x_max = torch.max(x, dim=-1)[0]
        torch.distributed.all_reduce(x_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_model_parallel_group())

        # Subtract the maximum value.
        x.sub_(x_max.unsqueeze(dim=-1))
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_x = x.exp()
        sum_exp_x = exp_x.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_x,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        softmax = torch.div(exp_x, sum_exp_x.unsqueeze(dim=-1))
        cls.save_for_backward(softmax)

        return softmax

    @staticmethod
    def backward(cls, grad_output):
        softmax, = cls.saved_tensors

        grad_input = softmax * grad_output - softmax * \
            torch.matmul(grad_output.unsqueeze(-2),
                         softmax.unsqueeze(-1)).squeeze(-1)

        return grad_input


class _ParallelLogSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(cls, x_in):
        pass

    @staticmethod
    def backward(cls, grad_output):
        pass


class _ParallelKLDivLoss(torch.autograd.Function):

    @staticmethod
    def forward(cls, p, q):
        
        pass

    @staticmethod
    def backward(cls, grad_output):
        pass


class _ParallelSoftCrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(cls, logits: torch.Tensor, targets: torch.Tensor):
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_model_parallel_group())
        # Subtract the maximum value.
        logits.sub_(logits_max.unsqueeze(dim=-1))
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        targets_max = torch.max(targets, dim=-1)[0]
        torch.distributed.all_reduce(targets_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_model_parallel_group())
        # Subtract the maximum value.
        targets.sub_(targets_max.unsqueeze(dim=-1))
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_targets = targets.exp()
        sum_exp_targets = exp_targets.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_targets,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        # targets_softmax: [b, s, v_p]
        targets_softmax = torch.div(exp_targets, sum_exp_targets.unsqueeze(-1))

        # sum_targets_softmax_logits: [b, s]
        sum_targets_softmax_logits = torch.matmul(
            targets_softmax.unsqueeze(-2), logits.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        torch.distributed.all_reduce(sum_targets_softmax_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        log_targets_softmax = torch.log(targets_softmax)
        sum_log_targets_softmax = torch.matmul(
            targets_softmax.unsqueeze(-2), log_targets_softmax.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        torch.distributed.all_reduce(sum_log_targets_softmax,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())


        loss = torch.log(sum_exp_logits) - sum_targets_softmax_logits + sum_log_targets_softmax

        logits_softmax = torch.div(exp_logits, sum_exp_logits.unsqueeze(-1))

        cls.save_for_backward(logits_softmax, targets_softmax)

        return loss

    @staticmethod
    def backward(cls, grad_output: torch.Tensor):
        logits_softmax, targets_softmax = cls.saved_tensors
        grad_input = (logits_softmax - targets_softmax) * grad_output.unsqueeze(-1)

        return grad_input, None


class _ParallelCrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(cls, logits_in, target):
        
        # Copy so the input remains unchanged.
        logits = logits_in.clone()
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_model_parallel_group())
        # Subtract the maximum value.
        logits.sub_(logits_max.unsqueeze(dim=-1))
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        # Get the partition's vocab indecies
        get_vocab_range = LogitUtility.logit_range_from_per_partition_logit_size
        partition_logit_size = logits_in.size()[-1]
        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        logit_start_index, logit_end_index = get_vocab_range(
            partition_logit_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < logit_start_index) | (target >= logit_end_index)
        # masked_target: local index of the target
        masked_target = target.clone() - logit_start_index
        # set the target that acturally from the other partition to 0
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = logits.view(-1, partition_logit_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                 device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d , masked_target_1d] # each postion select the target one
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(predicted_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        cls.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss


    @staticmethod
    def backward(cls, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = cls.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= (
            1.0 - target_mask.view(-1).float())

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None


class _ParallelMSELoss(torch.autograd.Function):
    
    @staticmethod
    def forward(cls, logits: torch.Tensor, targets: torch.Tensor):
        diff = logits - targets
        square_diff = diff * diff
        
        cls.save_for_backward(diff)

        output = square_diff.sum(-1)

        output = torch.distributed.all_reduce(output,
                                              op=torch.distributed.ReduceOp.SUM,
                                              group=get_model_parallel_group())

        return output

    @staticmethod
    def backward(cls, grad_output: torch.Tensor):
        diff, = cls.saved_tensors
        grad_input = 2 * diff * grad_output.unsqueeze(-1)
        return grad_input, None


class _ParallelCosineEmbeddingLoss(torch.autograd.Function):

    '''
    NOTE: Only for target = 1
    '''
    @staticmethod
    def forward(cls, logits_x: torch.Tensor, logits_y: torch.Tensor):
        dot_prod = (logits_x * logits_y).sum(-1)
        dot_prod = torch.distributed.all_reduce(dot_prod,
                                                op=torch.distributed.ReduceOp.SUM,
                                                group=get_model_parallel_group())

        len_x_sqrt = (logits_x * logits_x).sum(-1)
        len_x_sqrt = torch.distributed.all_reduce(len_x_sqrt,
                                             op=torch.distributed.ReduceOp.SUM,
                                             group=get_model_parallel_group())
        len_x = torch.sqrt(len_x_sqrt)

        len_y_sqrt = (logits_y * logits_y).sum(-1)
        len_y_sqrt = torch.distributed.all_reduce(len_y_sqrt,
                                             op=torch.distributed.ReduceOp.SUM,
                                             group=get_model_parallel_group())
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


def parallel_kl_div_loss(p, q):
    return _ParallelKLDivLoss.apply(p, q)


def parallel_cross_entropy_loss(logits, targets):
    return _ParallelCrossEntropyLoss.apply(logits, targets)


def parallel_soft_cross_entropy_loss(logits, targets):
    return _ParallelSoftCrossEntropyLoss.apply(logits, targets)


def parallel_softmax(p, q):
    return _ParallelSoftmax.apply(p, q)


def parallel_log_softmax(logits, targets):
    return _ParallelLogSoftmax.apply(logits, targets)


def parallel_mse_loss(logits, targets):
    return _ParallelMSELoss.apply(logits, targets)


def parallel_cos_loss(logits_x, logits_y):
    return _ParallelCosineEmbeddingLoss.apply(logits_x, logits_y)
