#include <algorithm>
#include <cassert>
#include <clocale>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "Source.hpp"
int run_normal();
int run_weibull();

using std::string;
using std::vector;

int main(){
#if defined(NORMAL)
    return run_normal();
#elif defined(WEIBULL)
    return run_weibull();
#else
#error "Не указано распределение"
#endif
}

//===========================Нормальное===========================
int run_normal() {
    try {
        setlocale(LC_ALL, "");

        const double conf = 0.95;
        Inp I = read_inp();

        // KM эмпирика
        vector<double> ycum, fcum;
        kaplan_meier(I.X, I.R, ycum, fcum);
        int m = (int)ycum.size();
        if (m < 3) throw std::runtime_error("Слишком мало событий (m<3)");

        // Дизайн-матрица X: [1, z_i], где z_i = Phi^{-1}(Blom)
        Mat X(m, 2);
        for (int i = 0; i < m; ++i) {
            double p = blom_p(i + 1, m);
            double z = normal_ppf(p);
            X(i, 0) = 1.0;
            X(i, 1) = z;
        }
        // Вектор отклика Y = x (в событиях)
        vector<double> Y = ycum;

        // β = (X'X)^(-1) X'Y
        Mat Xt = transpose(X);
        Mat XtX = mul(Xt, X);
        Mat XtX_inv = inv(XtX);
        vector<double> XtY = mul(Xt, Y);
        vector<double> beta(2, 0.0); // [a, s]
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) beta[j] += XtX_inv(j, k) * XtY[k];
        }

        // RSS и s^2
        vector<double> Yhat(m, 0.0);
        for (int i = 0; i < m; ++i) Yhat[i] = beta[0] + beta[1] * X(i, 1);
        double rss = 0; for (int i = 0; i < m; ++i) { double e = Y[i] - Yhat[i]; rss += e * e; }
        int dof = std::max(1, m - 2);
        double s2 = rss / dof;

        // Доверительные полосы для средней кривой на сетке P: v = [1, zq]^T
        double tcrit = student_t_ppf(0.5 * (1.0 + conf), dof);
        vector<double> XhatP(I.kp), XlowP(I.kp), XupP(I.kp);
        for (int i = 0; i < I.kp; ++i) {
            double zq = normal_ppf(I.P[i]);
            // se_mean = sqrt( s2 * v^T (X'X)^(-1) v )
            double v0 = 1.0, v1 = zq;
            double quad = v0 * (XtX_inv(0, 0) * v0 + XtX_inv(0, 1) * v1)
                + v1 * (XtX_inv(1, 0) * v0 + XtX_inv(1, 1) * v1);
            double se = std::sqrt(s2 * (1.0 + quad));
            double xhat = beta[0] + beta[1] * zq;
            XhatP[i] = xhat;
            XlowP[i] = xhat - tcrit * se;
            XupP[i] = xhat + tcrit * se;
        }

        // Серии для .xout: эмпирика в бумажных координатах, линии по сетке P
        vector<double> X_emp(m), Y_emp(m);
        for (int i = 0; i < m; ++i) { X_emp[i] = ycum[i]; Y_emp[i] = 5.0 + normal_ppf(fcum[i]); }
        vector<double> Y_grid(I.kp); for (int i = 0; i < I.kp; ++i) Y_grid[i] = 5.0 + normal_ppf(I.P[i]);

        vector<vector<double>> Xs = { X_emp, XlowP, XhatP, XupP };
        vector<vector<double>> Ys = { Y_emp, Y_grid, Y_grid, Y_grid };

        write_xout("Out/MLS_Normal.xout", Xs, Ys);
        write_out("Out/MLS_Normal.out", beta, s2, dof, I.n, m, I.P, XhatP, XlowP, XupP);

        std::cout << "Готово: Out/MLS_Normal.{out,xout}\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << "\n";
        return 1;
    }
    return 0;
}


//==========================Вейбулла===============
int run_weibull() {
    try {
        setlocale(LC_ALL, "");

        const double conf = 0.95;
        Inp I = read_inp_weibull();

        // KM
        vector<double> ycum, fcum;
        kaplan_meier(I.X, I.R, ycum, fcum);
        int m = (int)ycum.size();
        if (m < 3) throw std::runtime_error("Слишком мало событий (m<3)");
        for (double v : ycum) if (!(v > 0)) throw std::runtime_error("Вейбулл: x должны быть >0");

        // Дизайн-матрица X: [1, W_i], где W_i = ln(-ln(1 - F_KM(x_i)))
        Mat X(m, 2);
        vector<double> L(m);
        for (int i = 0; i < m; ++i) {
            double W = gumbel_y(fcum[i]);
            X(i, 0) = 1.0;
            X(i, 1) = W;
            L[i] = std::log(ycum[i]); // Y
        }

        // β = (X'X)^(-1) X'Y
        Mat Xt = transpose(X);
        Mat XtX = mul(Xt, X);
        Mat XtX_inv = inv(XtX);
        vector<double> XtY = mul(Xt, L);
        vector<double> beta(2, 0.0); // [alpha, beta]
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) beta[j] += XtX_inv(j, k) * XtY[k];
        }

        // RSS, s2
        vector<double> Lhat(m, 0.0);
        for (int i = 0; i < m; ++i) Lhat[i] = beta[0] + beta[1] * X(i, 1);
        double rss = 0; for (int i = 0; i < m; ++i) { double e = L[i] - Lhat[i]; rss += e * e; }
        int dof = std::max(1, m - 2);
        double s2 = rss / dof;

        // Средние доверительные полосы на сетке P (в лог-шкале!)
        double tcrit = student_t_ppf(0.5 * (1.0 + conf), dof);
        vector<double> XhatP(I.kp), XlowP(I.kp), XupP(I.kp);
        for (int i = 0; i < I.kp; ++i) {
            double wp = gumbel_y(I.P[i]);
            double v0 = 1.0, v1 = wp;
            double quad = v0 * (XtX_inv(0, 0) * v0 + XtX_inv(0, 1) * v1)
                + v1 * (XtX_inv(1, 0) * v0 + XtX_inv(1, 1) * v1);
            double se = std::sqrt(s2 * (1.0 + quad));
            double lnx = beta[0] + beta[1] * wp;
            double lnL = lnx - tcrit * se;
            double lnU = lnx + tcrit * se;
            XhatP[i] = std::exp(lnx);
            XlowP[i] = std::exp(lnL);
            XupP[i] = std::exp(lnU);
        }

        // Серии для .xout
        vector<double> X_emp(m), Y_emp(m);
        for (int i = 0; i < m; ++i) { X_emp[i] = ycum[i]; Y_emp[i] = 5.0 + gumbel_y(fcum[i]); }
        vector<double> Y_grid(I.kp); for (int i = 0; i < I.kp; ++i) Y_grid[i] = 5.0 + gumbel_y(I.P[i]);

        vector<vector<double>> Xs = { X_emp, XlowP, XhatP, XupP };
        vector<vector<double>> Ys = { Y_emp, Y_grid, Y_grid, Y_grid };

        write_xout_weibull("Out/MLS_Weibull.xout", Xs, Ys);
        write_out_weibull("Out/MLS_Weibull.out", beta, s2, dof, I.n, m, I.P, XhatP, XlowP, XupP);

        std::cout << "Готово: Out/MLS_Weibull.{out,xout}\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
