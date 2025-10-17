import torch
from torch.optim import Optimizer
from torch.optim._functional import sgd
from typing import Optional
from torch import Tensor
from typing import Union
from typing import Iterable
import copy


class paraflow(Optimizer):
    """Implementation of ParaflowS optimizer. It's based on two operators: a coarse
    solver and a fine solver. The coarse solver makes large steps to quickly
    approach the minimum, while the fine solver makes small steps to refine the
    solution. The optimizer alternates between these two solvers to efficiently
    minimize the loss function.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr_coarse (float, optional): learning rate of coarse solver (default: 1e-3)
        n_fine (int, optional): number of fine steps per iteration (default: 100)
        n_coarse (int, optional): maximum number of coarse steps per iteration (default: 200)
        momentum (float, optional): momentum factor (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing (default: False)
        foreach (bool, optional): whether to use foreach optimizations (default: None)
        differentiable (bool, optional): whether to record operations on the optimization step for higher order gradients (default: False)
        fused (bool, optional): whether to use fused optimizations (default: None)
        verbose (bool, optional): verbosity (default: False)

        NOTE: The step of both and fine solvers is based on the standard GD optimizer. So by now doesn't support momentum, weight decay and nesterov.
        NOTE: n_fine is fixed as 1/lr_coarse in the optimizer.py file
    """
    def __init__(self,
                 params: Iterable[Tensor],
                 lr_coarse: Union[float, Tensor] = 1e-3,
                 n_fine = 100, 
                 n_coarse = 200,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False,
                 *,                     # python syntax to force the use of keywords after this
                 maximize: bool = False,
                 foreach: Optional[bool] = None,
                 differentiable: bool = False,
                 fused: Optional[bool] = None,
                 verbose: bool = False):
        
        defaults = dict(
            lr_fine = lr_coarse / n_fine,
            n_fine = n_fine,
            lr_coarse = lr_coarse,
            n_coarse = n_coarse,
            momentum = momentum,
            dampening = dampening,
            weight_decay = weight_decay,
            nesterov = nesterov,
            maximize = maximize,
            foreach = foreach,
            differentiable = differentiable,
            fused = fused
        )

        self.lr_fine = lr_coarse / n_fine
        self.n_fine = n_fine
        self.lr_coarse = lr_coarse
        self.n_coarse = n_coarse
        self.verbose = verbose

        super(paraflow,self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=torch.zeros_like(p.data))

        #print('initialization of the class done with parameters:\n', defaults)

    def step(self,closure):
        # pass coarse
        loss_coarse, bff2 = self.coarse_solver(closure)
        loss_coarse = loss_coarse.item()

        # copy the state in bff1 for the correction step
        bff1 = safe_deepcopy(bff2)
        
        # pass fine
        loss_fine = self.fine_solver(closure).item()

        stay = True
        i = 0
        # correction cicle
        while stay and i<= self.n_coarse:

            for k,group in enumerate(self.param_groups):
                for j,p in enumerate(group["params"]):
                    
                    p.data = bff1[k]['params'][j] + p.data - bff2[k]['params'][j] # correction formula

            # coarse pass
            loss_coarse, bff1 = self.coarse_solver(closure)
            loss_coarse = loss_coarse.item()

            # check for the end of the cicle
            if loss_coarse > loss_fine and i > 0:
                stay = False
            else:
                loss_fine = loss_coarse
                i += 1

        # verbose print
        if self.verbose:
            print(f'ParaflowS iteration {i}, loss: {loss_fine}')

        return loss_fine

    def coarse_solver(self,closure):
        loss = closure()

        # copy the parameters to update without modify the class
        param_group_copy = safe_deepcopy(self.param_groups)

        for i,group in enumerate(self.param_groups):
            for j,p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                # forward pass
                d_p = p.grad
                param_group_copy[i]['params'][j] = param_group_copy[i]['params'][j] - group['lr_coarse'] * d_p

        return loss, param_group_copy
    
    def fine_solver(self,closure):
        # cicle over the n_fine stpes
        for i in range(self.n_fine):
            loss = closure()

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    # forward pass
                    d_p = p.grad
                    p.data = p.data - group["lr_fine"] * d_p
            
        return loss

def safe_deepcopy(obj):
    if isinstance(obj, torch.Tensor):
        # Clone tensor, break from computation graph
        return obj.detach().clone()
    elif isinstance(obj, dict):
        # Recursively copy each value
        return {k: safe_deepcopy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively copy list items
        return [safe_deepcopy(v) for v in obj]