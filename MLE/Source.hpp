#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <boost/math/distributions/normal.hpp>
#include <iomanip>
#include <limits>   
#include <cctype>   
#include <cstdlib>  
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>

using namespace std;

vector<vector<double>> cov_normal_asymp(vector <double>& x, vector <int>& r, double a, double s);
struct Sample { std::vector<double> x; std::vector<int> r; int n = 0; };

// --- Глобальные ссылки для обёртки цели NM ---
static const std::vector<double>* G_X = nullptr;
static const std::vector<int>* G_R = nullptr;

struct EmpiricalKM {
    vector<double> x_sorted;   // только наблюдаемые (r=0)
    vector<double> F_emp;      // соответствующие F (К–М)
};

// Простая К–М для правой цензуры I-типа.
// Вход: x — значения; r — флаги (0 — наблюдение, 1 — цензура).
inline EmpiricalKM kaplan_meier_Itype(const vector<double>& x,
    const vector<int>& r)
{
    const int n = static_cast<int>(x.size()); //n - размер x
    vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i; //заполняем массив индексами

    sort(idx.begin(), idx.end(),
        [&](int i, int j) { return x[i] < x[j]; }); // упорядочиваем по возрастанию x


    EmpiricalKM res;
    res.x_sorted.reserve(n); //выделяем память
    res.F_emp.reserve(n);

    double S = 1.0;   // выживаемость
    int at_risk = n;  // под риском


    for (int t = 0; t < n; ++t) {
        int i = idx[t];
        if (r[i] == 0) {
            // событие: d=1, S *= (at_risk - 1) / at_risk
            S *= double(at_risk - 1) / double(at_risk);
            const double F = 1.0 - S;

            res.x_sorted.push_back(x[i]);
            res.F_emp.push_back(F);
        }
        // убывает под-риском и при событии, и при цензуре
        at_risk -= 1;
    }
    return res;
}

// 1 / sqrt(2*pi) — числовое значение, чтобы не зависеть от M_PI
static const double INV_SQRT_2PI = 0.39894228040143267793994605993438;

inline double norm_pdf(double z) {
    return INV_SQRT_2PI * exp(-0.5 * z * z);
}

inline double norm_cdf(double z) {
    boost::math::normal_distribution<> N(0.0, 1.0);
    return cdf(N, z);
}

inline double norm_ppf(double p) {
    if (p <= 0.0) return -INFINITY;
    if (p >= 1.0) return  INFINITY;
    boost::math::normal_distribution<> N(0.0, 1.0);
    return quantile(N, p);
}

inline int neldermead(vector<double>& x0,
    double eps,
    double(*func)(vector<double>))
{
    const int n = static_cast<int>(x0.size());
    const int maxit = 200 * (n > 0 ? n : 1);

    const double rho = 1.0, chi = 2.0, psi = 0.5, sigma = 0.5;
    const double nonz = 0.05, z = 0.00025;

    vector<vector<double>> S(n + 1, x0);
    for (int k = 0; k < n; ++k) {
        vector<double> y = x0;
        y[k] = (y[k] != 0.0) ? (1.0 + nonz) * y[k] : z;
        S[k + 1] = y;
    }

    auto op = [&](const vector<double>& A,
        const vector<double>& B,
        double t)
        {
            vector<double> r(n);
            for (int i = 0; i < n; ++i) r[i] = A[i] + t * (B[i] - A[i]);
            return r;
        };

    int it = 0;
    while (it++ < maxit) {
        sort(S.begin(), S.end(),
            [&](const vector<double>& a, const vector<double>& b)
            { return func(a) < func(b); });

        double m = 0.0;
        for (const auto& s : S) m += func(s);
        m /= (n + 1);

        double v = 0.0;
        for (const auto& s : S) {
            double fs = func(s);
            v += (fs - m) * (fs - m);
        }
        if (sqrt(v / (n + 1)) < eps) break;

        vector<double> c(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                c[j] += S[i][j];
        for (double& cj : c) cj /= n;

        vector<double> xr = op(c, S[n], -rho);
        double fr = func(xr);

        if (fr < func(S[0])) {
            vector<double> xe = op(c, xr, chi);
            S[n] = (func(xe) < fr) ? xe : xr;
        }
        else if (fr < func(S[n - 1])) {
            S[n] = xr;
        }
        else {
            vector<double> xc =
                (fr < func(S[n])) ? op(c, xr, psi) : op(c, S[n], -psi);
            if (func(xc) <= min(fr, func(S[n]))) {
                S[n] = xc;
            }
            else {
                for (int i = 1; i <= n; ++i)
                    for (int j = 0; j < n; ++j)
                        S[i][j] = S[0][j] + sigma * (S[i][j] - S[0][j]);
            }
        }
    }

    x0 = S[0];
    return it;
}

// --- «скор»-цель ММП ---
static double normal_score_fn(vector<double> theta)
{
    const vector<double>& x = *G_X;
    const vector<int>& r = *G_R; // для оптимизации

    const double a = theta[0]; // a - среднее нормального распределения
    const double s = theta[1]; // s - стандартное отклонение
    if (!(s > 0.0)) return 1e12; // защита от негодных параметров

    double S1 = 0.0, S2 = 0.0; // сумматоры оценочных уравнений
    for (size_t i = 0; i < x.size(); ++i) {
        const double z = (x[i] - a) / s; // стандартизируем чтобы X ~ N(0,1)
        const double p = norm_cdf(z); // значение функции распределения
        const double d = norm_pdf(z); // значение плотности распределения
        const double psi = d / (1.0 - p); // хвостовой коэффициент 

        S1 += (1 - r[i]) * (x[i] - a) + r[i] * s * psi; // просто по формулам
        S2 += (1 - r[i]) * ((x[i] - a) * (x[i] - a) - s * s)
            + r[i] * s * s * (psi * ((x[i] - a) / s) - 1.0);
    }
    S1 /= double(x.size());
    S2 /= double(x.size());
    const double val = S1 * S1 + S2 * S2;

    if (!isfinite(val)) return 1e12;
    return val;
}

// Ковариационная матрица оценок  - ВОЗВРАЩАЕМ ВЕКТОР 2x2
inline vector<vector<double>> cov_normal_asymp(vector<double>& x,
    vector<int>& r,
    double a, double s)
{
    // Защита от нулевого/плохого s
    if (!(s > 0.0) || !std::isfinite(s)) s = 1.0;

    double J11 = 0.0, J12 = 0.0, J22 = 0.0;
    const double inv_s = 1.0 / s;
    const double inv_s2 = inv_s * inv_s;

    for (size_t i = 0; i < x.size(); ++i) {
        const double z = (x[i] - a) * inv_s;
        const double p = norm_cdf(z);
        const double d = norm_pdf(z);
        const double psi = d / (1.0 - p); // коэффициент Миллса для правой цензуры

        // Вклады в информацию по наблюдаемым и цензуре.
        // По размерности каждый вклад масштабируется 1/s^2.
        const double J11_i = (1 - r[i]) * (1.0) + r[i] * (psi * (psi - z));
        const double J12_i = r[i] * (psi * (z * (psi - z) - 1.0));
        const double J22_i = (1 - r[i]) * (2.0)
            + r[i] * (z * z * (psi * (psi - z)) - 2.0 * psi * z + 1.0);

        J11 += J11_i * inv_s2;
        J12 += J12_i * inv_s2;
        J22 += J22_i * inv_s2;
    }

    const double n = static_cast<double>(x.size());
    J11 /= n; J12 /= n; J22 /= n;

    double det = J11 * J22 - J12 * J12;
    if (!(det > 0.0) || !std::isfinite(det)) det = 1e-12;

    vector<vector<double>> V(2, vector<double>(2, 0.0));
    // V = J^{-1} — асимптотическая ковариация *на уровень одной наблюдаемости*
    V[0][0] = J22 / det;  V[0][1] = -J12 / det;
    V[1][0] = -J12 / det;  V[1][1] = J11 / det;
    return V;
}


// ММП Вейбулла
static pair<double, double> weibull_mle_2par(const vector<double>& x, const vector<int>& r)
{
    vector<double> obs; obs.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) if (r[i] == 0 && x[i] > 0.0) obs.push_back(x[i]);
    if (obs.size() < 2) throw runtime_error("Недостаточно наблюдений для ММП Вейбулла");

    vector<double> lx(obs.size());
    for (size_t i = 0; i < obs.size(); ++i) lx[i] = log(obs[i]);
    double m = accumulate(lx.begin(), lx.end(), 0.0) / double(lx.size());
    double s2 = 0.0; for (double v : lx) s2 += (v - m) * (v - m); s2 /= double(lx.size() - 1);
    double b = (s2 > 1e-12) ? 1.2 / sqrt(s2) : 2.0;
    if (b < 0.2)  b = 0.2;
    if (b > 20.0) b = 20.0;

    auto f = [&](double bb) {
        double A = 0.0, B = 0.0;
        for (double xi : obs) { double xb = pow(xi, bb); A += xb * log(xi); B += xb; }
        double lhs = 0.0; for (double xi : obs) lhs += log(xi);
        return lhs - A / B;
        };
    auto fp = [&](double bb) {
        double A = 0.0, B = 0.0, C = 0.0;
        for (double xi : obs) {
            double l = log(xi), xb = pow(xi, bb);
            A += xb * l; B += xb; C += xb * l * l;
        }
        double Aprime = C, Bprime = A;
        return -(Aprime * B - A * Bprime) / (B * B);
        };

    for (int it = 0; it < 50; ++it) {
        double val = f(b), der = fp(b);
        if (fabs(der) < 1e-12) break;
        double step = val / der;
        b -= step;
        if (b < 0.05) b = 0.05;
        if (fabs(step) < 1e-10) break;
    }
    double sum_xb = 0.0; for (double xi : obs) sum_xb += pow(xi, b);
    double c = pow(sum_xb / obs.size(), 1.0 / b);
    return { c,b };
}

// Оценки b,c по линейной регрессии (фолбэк, если ММП неудачен)
static std::pair<double, double> weibull_regression_fallback(const std::vector<double>& x,
    const std::vector<int>& r)
{
    // эмпирика
    auto emp = kaplan_meier_Itype(x, r);

    std::vector<double> X, Y;
    X.reserve(emp.x_sorted.size());
    Y.reserve(emp.x_sorted.size());

    // y = ln ln(1/(1-F)), x = ln(X), берём только 0 < F < 1
    for (size_t i = 0; i < emp.x_sorted.size(); ++i) {
        double F = emp.F_emp[i];
        if (F > 1e-6 && F < 1.0 - 1e-6 && emp.x_sorted[i] > 0.0) {
            X.push_back(std::log(emp.x_sorted[i]));
            Y.push_back(std::log(std::log(1.0 / (1.0 - F))));
        }
    }
    if (X.size() < 2) throw std::runtime_error("Слишком мало точек для регрессии Вейбулла");

    // обычная МНК-прямая: Y = a + b*X → оценка b — наклон, a — свободный член
    double Sx = 0, Sy = 0, Sxx = 0, Sxy = 0; const double n = double(X.size());
    for (size_t i = 0; i < X.size(); ++i) { Sx += X[i]; Sy += Y[i]; Sxx += X[i] * X[i]; Sxy += X[i] * Y[i]; }
    double den = n * Sxx - Sx * Sx;
    if (std::fabs(den) < 1e-14) throw std::runtime_error("Вырождение в регрессии Вейбулла");
    double b = (n * Sxy - Sx * Sy) / den;
    double a = (Sy - b * Sx) / n;

    // для Weibull: Y = b*ln x - b*ln c → a = -b ln c → c = exp(-a/b)
    double c = std::exp(-a / b);

    if (!std::isfinite(b) || !std::isfinite(c) || b <= 0.0 || c <= 0.0)
        throw std::runtime_error("Регрессия Вейбулла дала некорректные параметры");

    return { c, b };
}



// Грубая ковариация (2x2 через vector)
// Возвращаем ковариацию в координатах (p=ln c, q=1/b) и n_eff (число r=0)
static std::pair<std::vector<std::vector<double>>, int>
cov_weibull_asymp_eff(const std::vector<double>& x,
    const std::vector<int>& r,
    double c, double b)
{
    std::vector<double> obs; obs.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        if (r[i] == 0 && x[i] > 0.0) obs.push_back(x[i]);

    const int n_eff = static_cast<int>(obs.size());
    if (n_eff < 2) {
        std::vector<std::vector<double>> V_id(2, std::vector<double>(2, 0.0));
        V_id[0][0] = 1.0; V_id[1][1] = 1.0;
        return { V_id, std::max(n_eff, 1) };
    }

    const double p = std::log(c);
    const double q = 1.0 / b;

    double Jpp = 0.0, Jpq = 0.0, Jqq = 0.0;
    for (double xi : obs) {
        const double z = (std::log(xi) - p) / q;
        // Робастная аппроксимация наблюдённой информации (оставляем твою идею, но правильно нормируем):
        Jpp += 1.0;
        Jqq += 1.0 + std::fabs(z);
        Jpq += 0.1 * z;
    }

    const double neff = static_cast<double>(n_eff);
    Jpp /= neff; Jpq /= neff; Jqq /= neff;

    double det = Jpp * Jqq - Jpq * Jpq;
    if (!(det > 0.0) || !std::isfinite(det)) det = 1e-12;

    std::vector<std::vector<double>> V(2, std::vector<double>(2, 0.0));
    V[0][0] = Jqq / det;  V[0][1] = -Jpq / det;
    V[1][0] = -Jpq / det;  V[1][1] = Jpp / det;

    return { V, n_eff };
}



//=====================Утилиты для считывания=====================
static std::string low(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static bool next_label_is(std::istream& in, const char* a, const char* b = nullptr) {
    std::streampos pos = in.tellg();
    std::string s;
    if (!(in >> s)) return false;
    in.clear();
    in.seekg(pos);
    auto ls = low(s);
    return ls == low(a) || (b && ls == low(b));
}



static bool is_label(const std::string& t) {
    std::string s = low(t);
    return
        s == "n" || s == "samples_size" ||
        s == "beta" ||
        s == "eps" || s == "eps_output" ||
        s == "x" || s == "data" ||
        s == "r" || s == "censorizes" ||
        s == "kp" || s == "p" ||
        s == "step_of_minimization" || s == "lim_of_iteration";
}

static bool try_parse_double_loose(const std::string& tok, double& out) {
    // принимает и запятые, и точку; возвращает false, если это не число
    std::string t = tok;
    for (char& c : t) if (c == ',') c = '.';
    char* endp = nullptr;
    const double v = std::strtod(t.c_str(), &endp);
    if (endp && *endp == '\0') { out = v; return std::isfinite(out); }
    return false;
}

// прочитать ровно need чисел после текущей позиции, пропуская мусор,
// но останавливаясь с ошибкой на "реальной" метке секции.
static std::vector<double> read_numbers_loose(std::istream& in, int need, const char* ctx)
{
    std::vector<double> a; a.reserve(need);
    std::string tok;
    while ((int)a.size() < need && (in >> tok)) {
        // если это "настоящая" метка — значит в файле не хватает чисел для секции ctx
        if (is_label(tok))
            throw std::runtime_error(std::string("Недостаточно чисел в секции '") + ctx + "': встретили метку '" + tok + "'");
        double v;
        if (try_parse_double_loose(tok, v)) {
            a.push_back(v);
        }
        else {
            // мусорный токен — просто пропускаем
            continue;
        }
    }
    if ((int)a.size() < need)
        throw std::runtime_error(std::string("Недостаточно чисел в секции '") + ctx + "' (ожидалось " + std::to_string(need) + ")");
    return a;
}



// Читаем ровно N индикаторов 0/1:
//  - токены близкие к 0 → 0; близкие к 1 → 1;
//  - всё остальное игнорируем как мусор;
//  - если встретили "настоящую" метку раньше, чем набрали N, выдаём ошибку.
static std::vector<int> read_R_indicators(std::istream& in, int N) {
    std::vector<int> R; R.reserve(N);
    std::string tok;
    while ((int)R.size() < N && (in >> tok)) {
        if (is_label(tok)) {
            if ((int)R.size() < N)
                throw std::runtime_error("Недостаточно значений в секции 'R' — встретили метку '" + tok + "'");
            break;
        }
        double v;
        if (!try_parse_double_loose(tok, v)) continue; // мусор — пропустим
        if (v <= 0.25)       R.push_back(0);
        else if (v >= 0.75 && v <= 1.25) R.push_back(1);
        else {
            // явный мусор — просто пропустим
            continue;
        }
    }
    if ((int)R.size() < N)
        throw std::runtime_error("Недостаточно значений в секции 'R' (ожидалось " + std::to_string(N) + ")");
    return R;
}


// --- утилита: прочитать токен как double, заменяя запятую на точку
static double parse_double_token(string tok, const string& ctx) {
    for (char& c : tok) if (c == ',') c = '.';
    try {
        size_t p = 0;
        double v = stod(tok, &p);
        if (p != tok.size()) throw runtime_error("junk");
        if (!std::isfinite(v)) throw runtime_error("inf/nan");
        return v;
    }
    catch (...) {
        throw runtime_error("Ошибка чтения числа (" + ctx + "): '" + tok + "'");
    }
}





static int parse_int_token(string tok, const string& ctx) {
    for (char& c : tok) if (c == ',') c = '.';
    try {
        size_t p = 0;
        long long v = stoll(tok, &p);
        if (p != tok.size()) throw runtime_error("junk");
        if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
            throw runtime_error("range");
        return static_cast<int>(v);
    }
    catch (...) {
        throw runtime_error("Ошибка чтения целого (" + ctx + "): '" + tok + "'");
    }
}

// --- чтение Inp/<tag>.inp 
static Sample read_input_weibull(const std::string& tag)
{
    const std::string path = "Inp/" + tag + ".inp";
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Не найден файл " + path);

    Sample S{};
    std::string s, tok;

    // Samples_size | n
    if (!(in >> s)) throw std::runtime_error("Ожидалась метка Samples_size/n");
    if (low(s) != "samples_size" && low(s) != "n")
        throw std::runtime_error("Ожидалась Samples_size или n, получили '" + s + "'");
    if (!(in >> tok)) throw std::runtime_error("Ошибка чтения N");
    {
        double v; if (!try_parse_double_loose(tok, v)) throw std::runtime_error("Некорректный N");
        int N = (int)std::llround(v);
        if (N <= 1 || N > 1000000) throw std::runtime_error("Некорректный N (2..1e6)");
        S.n = N;
    }

    // beta
    if (!(in >> s)) throw std::runtime_error("Ожидалась метка beta");
    if (low(s) != "beta") throw std::runtime_error("Ожидалась beta");
    if (!(in >> tok)) throw std::runtime_error("Ошибка чтения beta");
    {
        double v; if (!try_parse_double_loose(tok, v)) throw std::runtime_error("Ошибка чтения beta");
        (void)v; // не используем
    }

    // step_of_minimization (ignore, если есть)
    if (next_label_is(in, "step_of_minimization")) {
        in >> s;                   // label
        if (!(in >> tok)) throw std::runtime_error("Ошибка чтения step_of_minimization");
        if (tok == ":" || tok == "=") { if (!(in >> tok)) throw std::runtime_error("Ошибка чтения step_of_minimization"); }
        // игнорируем значение
    }

    // eps_output | eps
    if (!(in >> s)) throw std::runtime_error("Ожидалась метка eps_output/eps");
    if (low(s) != "eps_output" && low(s) != "eps") throw std::runtime_error("Ожидалась eps_output/eps");
    if (!(in >> tok)) throw std::runtime_error("Ошибка чтения eps");
    {
        double v; if (!try_parse_double_loose(tok, v)) throw std::runtime_error("Ошибка чтения eps");
        (void)v;
    }

    // lim_of_iteration (ignore, если есть)
    if (next_label_is(in, "lim_of_iteration")) {
        in >> s;
        if (!(in >> tok)) throw std::runtime_error("Ошибка чтения lim_of_iteration");
        if (tok == ":" || tok == "=") { if (!(in >> tok)) throw std::runtime_error("Ошибка чтения lim_of_iteration"); }
    }

    // Data | X
    if (!(in >> s)) throw std::runtime_error("Ожидалась метка Data/X");
    if (low(s) != "data" && low(s) != "x") throw std::runtime_error("Ожидалась Data/X");
    {
        auto xs = read_numbers_loose(in, S.n, "X/Data");
        S.x = std::move(xs);
        for (int i = 0; i < S.n; ++i)
            if (!(S.x[i] >= 0.0)) throw std::runtime_error("X[" + std::to_string(i) + "] отрицательно");
    }

    // Censorizes | R
    if (!(in >> s)) throw std::runtime_error("Ожидалась метка Censorizes/R");
    if (low(s) != "censorizes" && low(s) != "r") throw std::runtime_error("Ожидалась Censorizes/R");
    S.r = read_R_indicators(in, S.n);

    // kp
    if (!(in >> s)) throw std::runtime_error("Ожидалась метка kp");
    if (!(in >> tok)) throw std::runtime_error("Ошибка чтения kp");
    int kp = 0;
    {
        double v; if (!try_parse_double_loose(tok, v)) throw std::runtime_error("Некорректный kp");
        kp = (int)std::llround(v); if (kp <= 0 || kp > 10000) throw std::runtime_error("Некорректный kp (1..10000)");
    }

    // P
    if (!(in >> s)) throw std::runtime_error("Ожидалась метка P");
    std::vector<double> P; P.reserve(kp);
    for (int i = 0; i < kp; ++i) {
        if (!(in >> tok)) throw std::runtime_error("Недостаточно значений P");
        double p; if (!try_parse_double_loose(tok, p) || !(p > 0.0 && p < 1.0))
            throw std::runtime_error("Некорректный P[" + std::to_string(i) + "]");
        P.push_back(p);
    }

    std::ofstream tmp("Out/" + tag + ".probs");
    if (!tmp) throw std::runtime_error("Не удалось открыть Out/" + tag + ".probs для записи");
    tmp << std::setprecision(17);
    for (double p : P) tmp << p << "\n";

    return S;
}




static Sample read_input_normal(const string& tag)
{
    ifstream in("Inp/" + tag + ".inp");
    if (!in) { throw runtime_error("Не найден файл Inp/" + tag + ".inp"); }

    string s; Sample S; int N; double beta, eps;
    in >> s >> N; S.n = N;
    in >> s >> beta;
    in >> s >> eps;
    in >> s; S.x.resize(N); for (int i = 0; i < N; i++) in >> S.x[i];
    in >> s; S.r.resize(N); for (int i = 0; i < N; i++) in >> S.r[i];
    in >> s; int kp; in >> kp;
    vector<double> P(kp); in >> s; for (int i = 0; i < kp; i++) in >> P[i];
    ofstream tmp("Out/" + tag + ".probs");
    if (!tmp) throw runtime_error("Не удалось открыть Out/" + tag + ".probs для записи");
    tmp << setprecision(17);
    for (double p : P) tmp << p << "\n";
    return S;
}