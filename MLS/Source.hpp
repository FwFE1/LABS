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

using std::string;
using std::vector;

struct Inp { int n = 0; vector<double> X; vector<int> R; int kp = 0; vector<double> P; };

/* ====== инимальная матрица (double) ====== */
struct Mat {
    int r{ 0 }, c{ 0 };
    vector<double> a;
    Mat() = default;
    Mat(int r_, int c_) : r(r_), c(c_), a(r_* c_, 0.0) {}
    double& operator()(int i, int j) { return a[i * c + j]; }
    double  operator()(int i, int j) const { return a[i * c + j]; }
    static Mat eye(int n) { Mat I(n, n); for (int i = 0; i < n; ++i) I(i, i) = 1.0; return I; }
};

static Mat transpose(const Mat& M) {
    Mat T(M.c, M.r);
    for (int i = 0; i < M.r; ++i) for (int j = 0; j < M.c; ++j) T(j, i) = M(i, j);
    return T;
}
static Mat mul(const Mat& A, const Mat& B) {
    if (A.c != B.r) throw std::runtime_error("Bad dims in mul");
    Mat C(A.r, B.c);
    for (int i = 0; i < A.r; ++i) {
        for (int k = 0; k < A.c; ++k) {
            double aik = A(i, k);
            for (int j = 0; j < B.c; ++j) C(i, j) += aik * B(k, j);
        }
    }
    return C;
}
static vector<double> mul(const Mat& A, const vector<double>& x) {
    if (A.c != (int)x.size()) throw std::runtime_error("Bad dims in mul Av");
    vector<double> y(A.r, 0.0);
    for (int i = 0; i < A.r; ++i) {
        double s = 0; for (int j = 0; j < A.c; ++j) s += A(i, j) * x[j];
        y[i] = s;
    }
    return y;
}
static Mat inv(Mat A) { // Gauss-Jordan
    if (A.r != A.c) throw std::runtime_error("inv: not square");
    int n = A.r;
    Mat I = Mat::eye(n);
    for (int col = 0; col < n; ++col) {
        // pivot
        int piv = col;
        double best = std::fabs(A(col, col));
        for (int i = col + 1; i < n; ++i) { double v = std::fabs(A(i, col)); if (v > best) { best = v; piv = i; } }
        if (best == 0) throw std::runtime_error("inv: singular");
        if (piv != col) {
            for (int j = 0; j < n; ++j) { std::swap(A(col, j), A(piv, j)); std::swap(I(col, j), I(piv, j)); }
        }
        // normalize row
        double d = A(col, col);
        for (int j = 0; j < n; ++j) { A(col, j) /= d; I(col, j) /= d; }
        // eliminate other rows
        for (int i = 0; i < n; ++i) {
            if (i == col) continue;
            double f = A(i, col);
            if (f == 0) continue;
            for (int j = 0; j < n; ++j) { A(i, j) -= f * A(col, j); I(i, j) -= f * I(col, j); }
        }
    }
    return I;
}

/* ====== утилиты вероятностей ====== */
static inline double normal_cdf(double x) { return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0))); }

static inline double normal_ppf(double p) {
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return  std::numeric_limits<double>::infinity();
    static const double a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    static const double b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const double d[] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };
    double q, r;
    if (p < 0.02425) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    else if (p > 1.0 - 0.02425) {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    else {
        q = p - 0.5; r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
}
static inline double student_t_ppf(double p, int nu) {
    double z = normal_ppf(p);
    if (nu <= 0) return z;
    double g = (z * z + 1.0) / (4.0 * nu);
    return z * (1.0 + g + 5.0 * z * z / (96.0 * nu * nu));
}

/* ====== KM (правое цензурирование) ====== */
static void kaplan_meier(const vector<double>& x, const vector<int>& r,
    vector<double>& ycum, vector<double>& fcum)
{
    int n = (int)x.size();
    vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return x[i] < x[j]; });

    double S = 1.0; int at_risk = n;
    for (int i = 0; i < n;) {
        double xi = x[idx[i]];
        int d = 0, c = 0; int j = i;
        while (j < n && x[idx[j]] == xi) { if (r[idx[j]] == 0) d++; else c++; ++j; }
        if (d > 0) {
            S *= (double)(at_risk - d) / (double)at_risk;
            ycum.push_back(xi);
            fcum.push_back(1.0 - S);
        }
        at_risk -= (d + c);
        i = j;
    }
}

/* ====== вспомогательное ====== */
static inline double blom_p(int i, int m) { return ((double)i - 0.375) / ((double)m + 0.25); }

static inline double gumbel_y(double p) {
    const double eps = 1e-12;
    p = std::min(std::max(p, eps), 1.0 - eps);
    return std::log(-std::log(1.0 - p));
}


static Inp read_inp() {
    Inp I;
    std::ifstream in("Inp/MLS_Normal.inp");
    if (!in) throw std::runtime_error("Не найден Inp/MLS_Normal.inp");
    string tag;
    in >> tag >> I.n;
    in >> tag; I.X.resize(I.n); for (int i = 0; i < I.n; ++i) in >> I.X[i];
    in >> tag; I.R.resize(I.n); for (int i = 0; i < I.n; ++i) in >> I.R[i];
    in >> tag >> I.kp;
    in >> tag; I.P.resize(I.kp); for (int i = 0; i < I.kp; ++i) in >> I.P[i];
    return I;
}

/* ====== запись файлов ====== */
static void write_xout(const string& out_path,
    const vector<vector<double>>& Xs,
    const vector<vector<double>>& Ys)
{
    std::ofstream f(out_path);
    if (!f) throw std::runtime_error("Не могу открыть: " + out_path);
    int nc = (int)Xs.size();
    f << nc << "\n";
    for (int i = 0; i < nc; ++i) { if (i) f << ' '; f << (int)Xs[i].size(); }
    f << "\n";
    for (int i = 0; i < nc; ++i) {
        for (size_t j = 0; j < Xs[i].size(); ++j) f << Xs[i][j] << (j + 1 == Xs[i].size() ? '\n' : ' ');
    }
    for (int i = 0; i < nc; ++i) {
        for (size_t j = 0; j < Ys[i].size(); ++j) f << Ys[i][j] << (j + 1 == Ys[i].size() ? '\n' : ' ');
    }
}
static void write_out(const string& out_path,
    const vector<double>& beta, double s2, int dof,
    int n, int m,
    const vector<double>& P,
    const vector<double>& Xhat,
    const vector<double>& Xlow,
    const vector<double>& Xup)
{
    std::ofstream f(out_path);
    if (!f) throw std::runtime_error("Не могу открыть: " + out_path);
    f.setf(std::ios::fixed); f << std::setprecision(10);
    f << "Method: MLS_Normal (matrix OLS)\n";
    f << "n=" << n << "\n";
    f << "k(events)=" << m << "\n";
    f << "a=" << beta[0] << "\n";
    f << "sigma=" << beta[1] << "\n";
    f << "s2=" << s2 << ", dof=" << dof << "\n";
    f << "P\n";     for (double p : P)    f << p << " "; f << "\n";
    f << "Xp\n";    for (double v : Xhat) f << v << " "; f << "\n";
    f << "Xlow\n";  for (double v : Xlow) f << v << " "; f << "\n";
    f << "Xup\n";   for (double v : Xup)  f << v << " "; f << "\n";
}

static Inp read_inp_weibull() {
    Inp I;
    std::ifstream in("Inp/MLS_Weibull.inp");
    if (!in) throw std::runtime_error("Не найден Inp/MLS_Weibull.inp");
    string tag;
    in >> tag >> I.n;
    in >> tag; I.X.resize(I.n); for (int i = 0; i < I.n; ++i) in >> I.X[i];
    in >> tag; I.R.resize(I.n); for (int i = 0; i < I.n; ++i) in >> I.R[i];
    in >> tag >> I.kp;
    in >> tag; I.P.resize(I.kp); for (int i = 0; i < I.kp; ++i) in >> I.P[i];
    return I;
}

static void write_xout_weibull(const string& out_path,
    const vector<vector<double>>& Xs,
    const vector<vector<double>>& Ys)
{
    std::ofstream f(out_path);
    if (!f) throw std::runtime_error("Не могу открыть: " + out_path);
    int nc = (int)Xs.size();
    f << nc << "\n";
    for (int i = 0; i < nc; ++i) { if (i) f << ' '; f << (int)Xs[i].size(); }
    f << "\n";
    for (int i = 0; i < nc; ++i) {
        for (size_t j = 0; j < Xs[i].size(); ++j) f << Xs[i][j] << (j + 1 == Xs[i].size() ? '\n' : ' ');
    }
    for (int i = 0; i < nc; ++i) {
        for (size_t j = 0; j < Ys[i].size(); ++j) f << Ys[i][j] << (j + 1 == Ys[i].size() ? '\n' : ' ');
    }
}
static void write_out_weibull(const string& out_path,
    const vector<double>& beta, double s2, int dof,
    int n, int m,
    const vector<double>& P,
    const vector<double>& Xhat,
    const vector<double>& Xlow,
    const vector<double>& Xup)
{
    std::ofstream f(out_path);
    if (!f) throw std::runtime_error("Не могу открыть: " + out_path);
    f.setf(std::ios::fixed); f << std::setprecision(10);
    f << "Method: MLS_Weibull (matrix OLS)\n";
    f << "n=" << n << "\n";
    f << "k(events)=" << m << "\n";
    f << "alpha_lin=" << beta[0] << "\n";
    f << "beta_lin=" << beta[1] << "\n";
    f << "lambda_hat=" << std::exp(beta[0]) << "\n";
    f << "k_hat=" << (1.0 / std::max(1e-300, beta[1])) << "\n";
    f << "s2=" << s2 << ", dof=" << dof << "\n";
    f << "P\n";     for (double p : P)    f << p << " "; f << "\n";
    f << "Xp\n";    for (double v : Xhat) f << v << " "; f << "\n";
    f << "Xlow\n";  for (double v : Xlow) f << v << " "; f << "\n";
    f << "Xup\n";   for (double v : Xup)  f << v << " "; f << "\n";
}