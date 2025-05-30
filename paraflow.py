import torch
from torch.optim import Optimizer
from torch.optim._functional import sgd
from typing import Optional
from torch import Tensor
from typing import Union
from typing import Iterable
import copy


class paraflow(Optimizer):
    def __init__(self,
                 params: Iterable[Tensor],
                 lr_coarse: Union[float, Tensor] = 1e-3,
                 n_fine = 100, 
                 n_coarse = 200,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False,
                 *,
                 maximize: bool = False,
                 foreach: Optional[bool] = None,
                 differentiable: bool = False,
                 fused: Optional[bool] = None):
        
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