#include <iostream>
#include <vector>

template<typename T>
class FNN{
    private:
        const std::vector<int> layer_size;
        const std::string activation;
        const std::string initializer;

        std::vector<T> params;

    public:
        FNN(std::vector<int>& ls,std::string Act ,std::string init):layer_size(ls),activation(Act), initializer(init){};
};