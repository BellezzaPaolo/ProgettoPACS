#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

#include "Optimizer.hpp"

class Gradient_Descent : public Optimizer{
    private:
        double lr;
        int batch_size;
    public:
        Gradient_Descent(std::function<double(Eigen::Matrix<double,2,1>&)> f, std::function<Eigen::Matrix<double,2,1>(Eigen::Matrix<double,2,1>&)> df, int max_iter, double tol,double LR, int batch):
        Optimizer(f,df,max_iter,tol),lr(LR),batch_size(batch){};

        virtual ~Gradient_Descent();

        inline double get_lr() const {return lr;}
        inline void set_lr(double LR) {lr = LR;}

        inline int get_batch_size() const {return batch_size;}
        inline void set_batch_size(int batch) {batch_size = batch;}

        void optimize() override;

};
#endif