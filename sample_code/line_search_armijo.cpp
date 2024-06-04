#include <iostream>
#include <Eigen/Dense>
#include <string>
#define INITIAL_STEP_SIZE 5

double objFunc(Eigen::VectorXd x);
Eigen::VectorXd getGrad(Eigen::VectorXd x);
void lineSearchArmijo(Eigen::VectorXd &x, double c = 0.2);

int main()
{
    int n = 4; // Must be an even number
    Eigen::VectorXd vec(n);
    vec<<3.1,3.2,4.3,5.4;
    // Eigen::VectorXd vec = Eigen::VectorXd::Random(n);
    // Eigen::VectorXd vec=Eigen::VectorXd::Zero(n);
    // Eigen::VectorXd vec=Eigen::VectorXd::Ones(n);
    lineSearchArmijo(vec);
    return 0;
}

double objFunc(Eigen::VectorXd x)
{
    double res = 0.0;
    int iter = x.size() / 2 - 1;
    for (int i = 0; i <= iter; i++)
    {
        double val1 = x(2 * i) * x(2 * i) - x(2 * i + 1);
        double val2 = (x(2 * i) - 1) * (x(2 * i) - 1);
        res += 100 * val1 * val1 + val2 * val2;
    }
    return res;
}

Eigen::VectorXd getGrad(Eigen::VectorXd x)
{
    Eigen::VectorXd res = Eigen::VectorXd::Zero(x.size());
    /*f(x) is a vector-to-scalar mapping:
    df/dx={
    df/dx_0,
    df/dx_1,
    ...
    df/dx_(n-1)
    Therefore, the gradient is a vector-to-vector mapping
    }*/
    int iter = x.size() - 1;
    for (int j = 0; j <= iter; j++)
    {
        if (j % 2 == 0)
            res(j) = (200 * (x(j) * x(j) - x(j + 1)) + 1) * 2 * x(j) - 2;
        else
            res(j) = -200 * (x(j - 1) * x(j - 1) - x(j));
    }

    return res;
}

void lineSearchArmijo(Eigen::VectorXd &x, double c)
{   
    Eigen::VectorXd x_now = x;
    Eigen::VectorXd grad = getGrad(x_now);
    Eigen::VectorXd d = -grad;
    while(d.norm()>0.01)
    {
        double opt_step=INITIAL_STEP_SIZE;
        while (objFunc(x_now + opt_step * d) > objFunc(x_now) + c * opt_step * d.transpose() * grad)
        {
            opt_step = opt_step / 2;
        }
        x_now = x_now + opt_step * d;
        grad = getGrad(x_now);
        d = -grad;
        std::cout << "Step value: "<< opt_step <<"Iter x: \n"<<x_now << std::endl;
    }
    x = x_now;
}