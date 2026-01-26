# Libtorch PINN Guide (C++ only)

This guide explains how to rewrite the current PINN training flow using **libtorch (PyTorch C++ API)** while keeping the **training loop in C++** and supporting a **Laplacian-based PDE loss**.

The goal is to compute a loss like

$$
L = \frac{1}{N}\sum_{k=1}^N (\Delta u_\theta(x_k) - f(x_k))^2
$$

and obtain correct gradients $\nabla_\theta L$ for your optimizer (e.g. Paraflow), entirely in C++.

---

## 0) Why libtorch for Laplacian-loss training

- You need **second derivatives w.r.t. inputs** ($\Delta u$) and **first derivatives w.r.t. parameters** ($\nabla_\theta L$).
- Libtorch autograd supports higher-order derivatives by keeping the graph with `create_graph=true`.
- You can still run **your own optimizer**: compute grads with autograd, then update parameters manually.

---

## 1) Install LibTorch

1. Download libtorch from the official PyTorch site (C++ distribution):
   - CPU-only (simplest)
   - or CUDA build (if you want GPU)
2. Unzip it somewhere stable, e.g.:

- `~/libtorch`

You will use this path in CMake.

---

## 2) Add libtorch to CMake

In your top-level `CMakeLists.txt`, add something like:

```cmake
# Point this to your extracted libtorch folder
set(CMAKE_PREFIX_PATH "/home/paolo/libtorch")

find_package(Torch REQUIRED)

add_executable(pinn_torch pinn_torch.cpp)

target_link_libraries(pinn_torch PRIVATE ${TORCH_LIBRARIES})
target_compile_features(pinn_torch PRIVATE cxx_std_17)

# Optional but common with libtorch
set_property(TARGET pinn_torch PROPERTY CXX_STANDARD 17)
```

Build:

```bash
mkdir -p build && cd build
cmake ..
cmake --build . -j
```

---

## 3) Define the network as a `torch::nn::Module`

Create a file like `pinn_torch.cpp` and define an MLP.

Minimal example (2D input, scalar output):

```cpp
#include <torch/torch.h>

struct MLPImpl : torch::nn::Module {
    torch::nn::Sequential seq;

    MLPImpl(int in_dim, int hidden, int depth, int out_dim) {
        seq->push_back(torch::nn::Linear(in_dim, hidden));
        seq->push_back(torch::nn::Tanh());
        for(int i = 1; i < depth; ++i){
            seq->push_back(torch::nn::Linear(hidden, hidden));
            seq->push_back(torch::nn::Tanh());
        }
        seq->push_back(torch::nn::Linear(hidden, out_dim));
        register_module("seq", seq);
    }

    torch::Tensor forward(torch::Tensor x) {
        return seq->forward(x);
    }
};
TORCH_MODULE(MLP);
```

---

## 4) Compute Laplacian with autograd (core PINN step)

Assume:
- `x` is a tensor of shape `[N, 2]`
- `u = net(x)` is shape `[N, 1]`

### 4.1 Enable gradients on inputs

```cpp
x = x.clone().set_requires_grad(true);
```

### 4.2 First derivative $\nabla_x u$

Compute gradient of `u.sum()` w.r.t `x`:

```cpp
auto u = net->forward(x);           // [N, 1]
auto ones = torch::ones_like(u);    // [N, 1]

// du/dx: [N, 2]
auto du_dx = torch::autograd::grad(
    /*outputs=*/{u},
    /*inputs=*/{x},
    /*grad_outputs=*/{ones},
    /*retain_graph=*/true,
    /*create_graph=*/true
)[0];
```

### 4.3 Second derivatives (diagonal of Hessian)

For Laplacian in 2D, you only need $\partial^2 u/\partial x_0^2$ and $\partial^2 u/\partial x_1^2$.

```cpp
auto du_dx0 = du_dx.index({torch::indexing::Slice(), 0}); // [N]
auto du_dx1 = du_dx.index({torch::indexing::Slice(), 1}); // [N]

auto d2u_dx02 = torch::autograd::grad(
    {du_dx0}, {x}, {torch::ones_like(du_dx0)}, true, true
)[0].index({torch::indexing::Slice(), 0}); // [N]

auto d2u_dx12 = torch::autograd::grad(
    {du_dx1}, {x}, {torch::ones_like(du_dx1)}, true, true
)[0].index({torch::indexing::Slice(), 1}); // [N]

auto lap = d2u_dx02 + d2u_dx12; // [N]
```

---

## 5) PDE loss and backprop

Example PDE residual:

```cpp
auto f = torch::zeros_like(lap);               // replace with your rhs f(x)
auto residual = lap - f;                       // [N]
auto loss = torch::mean(residual * residual);  // scalar

loss.backward();
```

After `backward()`, every parameter has `param.grad()`.

---

## 6) Use Paraflow (your optimizer) in C++

You have two common integration patterns:

### Option A: Paraflow updates torch parameters directly

- After `loss.backward()`:
  - read grads from `param.grad()`
  - apply your Paraflow update rule to `param.data()`

This is the simplest if you already implement Paraflow as “parameter vector update”.

### Option B: Paraflow owns a flat vector $\theta$

- Keep a flat vector `theta` in Eigen/Std.
- Each iteration:
  1. copy `theta` into module parameters
  2. compute loss + grads
  3. copy grads back into an Eigen/Std vector
  4. update `theta` with Paraflow

This is usually easiest if Paraflow expects a single flat vector.

---

## 7) Practical tips

- Use `create_graph=true` for first derivatives; otherwise second derivatives will be zero / impossible.
- Start with small networks and small batch sizes; higher-order autograd can be heavy.
- Use `torch::NoGradGuard` during parameter updates (so updates don’t become part of the graph).

Example:

```cpp
{
  torch::NoGradGuard ng;
  for (auto& p : net->parameters()) {
    p -= lr * p.grad();
    p.grad().zero_();
  }
}
```

---

## 8) Minimal execution skeleton

```cpp
int main(){
  torch::manual_seed(0);

  MLP net(2, 64, 4, 1);

  // Example collocation points
  auto x = torch::rand({128, 2}, torch::kDouble);
  x = x.clone().set_requires_grad(true);

  auto u = net->forward(x);
  auto du_dx = torch::autograd::grad({u}, {x}, {torch::ones_like(u)}, true, true)[0];

  auto du_dx0 = du_dx.index({torch::indexing::Slice(), 0});
  auto du_dx1 = du_dx.index({torch::indexing::Slice(), 1});

  auto d2u_dx02 = torch::autograd::grad({du_dx0}, {x}, {torch::ones_like(du_dx0)}, true, true)[0]
                    .index({torch::indexing::Slice(), 0});
  auto d2u_dx12 = torch::autograd::grad({du_dx1}, {x}, {torch::ones_like(du_dx1)}, true, true)[0]
                    .index({torch::indexing::Slice(), 1});

  auto lap = d2u_dx02 + d2u_dx12;
  auto loss = torch::mean(lap * lap);
  loss.backward();

  // Replace this block with Paraflow update
  {
    torch::NoGradGuard ng;
    double lr = 1e-3;
    for (auto& p : net->parameters()) {
      p -= lr * p.grad();
      p.grad().zero_();
    }
  }

  std::cout << "loss=" << loss.item<double>() << "\n";
}
```

---

## Next step (if you want)

If you tell me:
- CPU-only or CUDA
- where you’ll put libtorch (path)

…I can write a *minimal* `pinn_torch.cpp` example that matches your current PDE interface and outputs a flat gradient vector compatible with Paraflow.
