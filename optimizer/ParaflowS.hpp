#ifndef PARAFLOWS_HPP
#define PARAFLOWS_HPP

#include "Optimizer.hpp"

class ParaflowS : public Optimizer{
    private:
        double lr_coarse = lr_fine * n_fine;
        double lr_fine;
        int n_fine;
        int n_coarse;
        int batch_size_coarse;
        int batch_size_fine;

        Eigen::Matrix<double,2,1> coarse_parameters;
        Eigen::Matrix<double,2,1> old_correction;
        Eigen::Matrix<double,2,1> correction;

        bool stay = true;
    
    public:
        ParaflowS(std::function<double(Eigen::Matrix<double,2,1>&)> f, std::function<Eigen::Matrix<double,2,1>(Eigen::Matrix<double,2,1>&)> df, int max_iter, double tol,
        double LR_F,int N_fine, int N_coarse,int BS_coarse, int BS_fine):
        Optimizer(f,df,max_iter,tol),lr_fine(LR_F),n_fine(N_fine),n_coarse(N_coarse),batch_size_coarse(BS_coarse),batch_size_fine(BS_fine){};

        double get_lr_coarse() const {return lr_coarse;}
        void set_lr_coarse(double LR) {lr_coarse = LR;}

        double get_lr_fine() const {return lr_fine;}
        void set_lr_fine(double LR) {lr_fine = LR;}

        int get_n_fine() const {return n_fine;}
        void set_n_fine(int N_fine) {n_fine = N_fine;}

        int get_n_coarse() const {return n_coarse;}
        void set_n_coarse(int N_coarse) {n_coarse = N_coarse;}

        int get_batch_size_coarse() const {return batch_size_coarse;}
        void set_batch_size_coarse(int batch) {batch_size_coarse = batch;}

        int get_batch_size_fine() const {return batch_size_fine;}
        void set_batch_size_fine(int batch) {batch_size_fine = batch;}

        void optimize() override;

        double coarse_solver();

        double fine_solver();

};
#endif