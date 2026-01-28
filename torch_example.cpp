#include <iostream>
#include <chrono>
#include <torch/torch.h>

#include "pinn_torch.hpp"

int main()
{
    torch::manual_seed(0);

    constexpr Initializer_weight Iw = Initializer_weight::Uniform;
    constexpr Initializer_bias Ib = Initializer_bias::Normal;
    constexpr Activation A = Activation::Tanh;

    FNNImpl<A> net(std::vector<int64_t>{2, 32, 32, 1});
    net.to(torch::kDouble);
    net.initialize<Iw,Ib>();

    net.print();

    // Optimizer
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));

    const int64_t epochs = 500;
    torch::Tensor x, y, target, loss;

    auto t_start = std::chrono::high_resolution_clock::now();

    for(int64_t epoch = 0; epoch < epochs; ++epoch){
        // Input batch
        x = torch::rand({5, 2}, torch::dtype(torch::kDouble));

        // Dummy target for demonstration (here: all ones)
        target = torch::ones({5, 1}, torch::dtype(torch::kDouble));

        // Forward pass
        y = net.forward(x);

        // Compute loss (Mean Squared Error)
        loss = torch::mse_loss(y, target);

        // Backward pass
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if(epoch % 50 == 0){
            std::cout << "Epoch " << epoch
                      << ", loss = " << loss.item<double>() << '\n';
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    std::cout << "\nFinal batch x:\n" << x << "\n\n";
    std::cout << "Final batch y:\n" << y << "\n\n";
    std::cout << "Final loss: " << loss.item<double>() << "\n";
    std::cout << "\nTraining time: " << elapsed << " ms\n";

    return 0;
}
