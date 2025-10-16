#include "Optimizer.hpp"

#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

class Gradient_Descent : public Optimizer{
    private:
        double lr;
        int batch_size;
    public:
        Gradient_Descent(std::function<double(std::vector<double>&)> f, std::function<std::vector<double>(std::vector<double>&)> df, int max_iter, double tol,int dim,double LR, int batch):
        Optimizer(f,df,max_iter,tol,dim),lr(LR),batch_size(batch){};

        double get_lr() const {return lr;}
        void set_lr(double LR) {lr = LR;}

        int get_batch_size() const {return batch_size;}
        void set_batch_size(int batch) {batch_size = batch;}

        void optimize() override{};

};
#endif