import torch
from torch.optim import Optimizer
from torch.optim._functional import sgd
from typing import Optional
from torch import Tensor
from typing import Union
from typing import Iterable

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
    """
    def __init__(self,
                 params: Iterable[Tensor],
                 lr_fine: Union[float, Tensor] = 1e-3,
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
                 fused: Optional[bool] = None,
                 verbose: bool = False):
        
        defaults = dict(
            lr_fine = lr_fine,
            n_fine = n_fine,
            batch_fine = None,
            lr_coarse = lr_fine * n_fine,
            n_coarse = n_coarse,
            batch_coarse = None,
            momentum = momentum,
            dampening = dampening,
            weight_decay = weight_decay,
            nesterov = nesterov,
            maximize = maximize,
            foreach = foreach,
            differentiable = differentiable,
            fused = fused,
            coarse = True
        )

        self.lr_fine = lr_fine
        self.n_fine = n_fine
        self.batch_fine = None
        self.lr_coarse = lr_fine * n_fine
        self.n_coarse = n_coarse
        self.batch_coarse = None
        self.verbose = verbose
        self.coarse = True
        self.callbacks = None
        self.stop_iteration = False

        self.counter = dict(call_coarse = 0, call_fine = 0, correction_steps = 0, iterations = 0) 

        super(paraflow,self).__init__(params, defaults)
        
        for group in self.param_groups:
            for p in group['params']:
                # store detached clones to avoid tracking graph and aliasing
                self.state[p] = dict(
                    bff1=torch.zeros_like(p.data).detach(),
                    correction=torch.zeros_like(p.data).detach(),
                    U_coarse=torch.zeros_like(p.data).detach(),
                )

        #print('initialization of the class done with parameters:\n', defaults)

    def step(self,closure):
        self.counter['iterations'] += 1
        # pass coarse
        loss_coarse = self.coarse_solver(closure).item()
        # print(f'coarse loss: {loss_coarse}')

        # pass fine
        loss_fine = self.fine_solver(closure).item()
        # print(f'after fine loss: {loss_fine}')

        stay = True
        i = 0
        L = 0
        # correction cicle
        while stay and i < self.n_coarse and not self.stop_iteration:

            if getattr(self, "callbacks", None) is not None:
                try:
                    self.callbacks.on_epoch_begin()
                    self.callbacks.on_batch_begin()
                except Exception:
                    pass

            bool1 = 0
            bool2 = 0
            diff = 0.0
            for k,group in enumerate(self.param_groups):
                for j,p in enumerate(group["params"]):
                    if i == 0:
                        L += 1
                        # use detached clones to avoid unexpected aliasing
                        # self.state[p]['U_fine'] = p.data.clone().detach()
                        self.state[p]['correction'].copy_(p.data - self.state[p]['bff1'])

                    # update parameter in-place to avoid changing the Parameter object's data pointer
                    #else:
                    p.data.copy_(self.state[p]['bff1'] + self.state[p]['correction'])

                    # diff += torch.norm(self.state[p]['U_fine'] - p.data).item()
                    
                    # use allclose for floating point comparison
                    # if torch.allclose(self.state[p]['bff1'], self.state[p]['bff2']):
                    #     bool2 += 1
            # if bool1 == L:
            #     print('ufine == data')
            # if bool2 == L:
                # # print('bff1 == bff2')

            # print(f'iter {i} | U_fine - data| = {diff}')
            # coarse pass
            loss_coarse = self.coarse_solver(closure).item()
            
            #print(f'ParaflowS correction step {i}, loss: {loss_fine} --> {loss_coarse}')
            # check for the end of the cicle
            if loss_coarse <= loss_fine or i == 0:
                #print('copia')
                for group in self.param_groups:
                    for p in group["params"]:
                        self.state[p]["U_coarse"] = p.data.clone().detach()
                loss_fine = loss_coarse
                i += 1
            else:
                # print(f'Stopping correction steps at {i} iterations')
                self.counter["correction_steps"] += i
                for group in self.param_groups:
                    for p in group["params"]:
                        p.data = self.state[p]["U_coarse"].clone().detach()
                stay = False

            if getattr(self, "callbacks", None) is not None:
                try:
                    self.callbacks.on_batch_end()
                    self.callbacks.on_epoch_end()
                except Exception:
                    pass

        # verbose print
        if self.verbose:
            print(f'ParaflowS iteration {i}, loss: {loss_fine}')

        return loss_fine

    def coarse_solver(self,closure):
        loss = None
        with torch.enable_grad():
            loss = closure()

        self.counter['call_coarse'] += 1

        for i,group in enumerate(self.param_groups):
            for j,p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                # forward pass
                d_p = p.grad
                # store detached clone to avoid linking to the computational graph
                self.state[p]['bff1'].copy_(p.data - group['lr_coarse'] * d_p)
        #print(loss.item())

        return loss
    

    def fine_solver(self,closure):
        loss = None
        self.counter['call_fine'] += self.n_fine
        
        # cicle over the n_fine stpes
        for i in range(self.n_fine):

            if getattr(self, "callbacks", None) is not None:
                try:
                    self.callbacks.on_epoch_begin()
                    self.callbacks.on_batch_begin()
                except Exception:
                    pass
            
            with torch.enable_grad():
                loss = closure()

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    # forward pass
                    d_p = p.grad
                    # update in-place to preserve storage and avoid creating new tensor
                    with torch.no_grad():
                        # p.data += (-group["lr_fine"]) * d_p
                        p.data.add_(d_p, alpha=-group["lr_fine"])
            
            if getattr(self, "callbacks", None) is not None:
                try:
                    self.callbacks.on_batch_end()
                    self.callbacks.on_epoch_end()
                except Exception:
                    pass

            if self.stop_iteration:
                break
            
        return loss