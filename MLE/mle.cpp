#include <limits>   
#include <cctype>   
#include <cstdlib>  
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include "Source.hpp"

#undef max //без этого выдает ошибку

int run_normal();
int run_weibull();


using namespace std;

int main(int argc, const char** argv){
    setlocale(LC_ALL, "");      
    setlocale(LC_NUMERIC, "C"); 
#ifdef NORMAL
    return run_normal();
#elif WEIBULL
    return run_weibull();
//#else
//#error "Не выбрано распределение"
#endif
}


// Структура выборки

//==================================Нормальное===============================
int run_normal() {
    const string TAG = "MLE_Normal";
    auto S = read_input_normal(TAG);

    // Начальные по наблюдаемым (r=0)
    vector<double> obs;
    obs.reserve(S.n);
    for (int i = 0; i < S.n; i++) if (S.r[i] == 0) obs.push_back(S.x[i]);

    // Начальные приближения a0, s0 из наблюдаемых
    double a0 = 0.0, s0 = 1.0;
    if (!obs.empty()) { //случай, если есть без цензуры
        a0 = accumulate(obs.begin(), obs.end(), 0.0) / double(obs.size()); // мат. ожидание (среднее)
        double q = 0.0;
        for (double v : obs) q += (v - a0) * (v - a0); 
        // на старт оставим несмещённую, но далее ниже для "без цензуры" исправим на MLE
        s0 = (obs.size() > 1) ? sqrt(q / double(obs.size() - 1)) : 1.0;
    }
    else { //случай, если наблюдаемых нет
        vector<double> xs = S.x; sort(xs.begin(), xs.end()); //сортируем по возрастанию
        a0 = xs[xs.size() / 2]; //медиана как стартовая точка
        s0 = max(1e-3, (xs.back() - xs.front()) / 6.0); //(макс-мин)/6 просто удобно
    }

    vector<double> theta{ a0, max(1e-6,s0) }; //чтобы соответствовало значениеям
    const bool censored = ((int)obs.size() != S.n); //флаг цензуры

    if (censored) {
        G_X = &S.x;
        G_R = &S.r;
        neldermead(theta, 1e-10, &normal_score_fn);
    }

    // Оценки параметров (выбираем финальное мат.ожидание и СКО)
    double a_hat, s_hat;
    if (!censored) {
        // Без цензуры - берём истинные ММП: среднее и s с делением на n
        a_hat = accumulate(S.x.begin(), S.x.end(), 0.0) / double(S.n);
        double q = 0.0;
        for (double v : S.x) q += (v - a_hat) * (v - a_hat);
        s_hat = (S.n > 0) ? sqrt(q / double(S.n)) : theta[1];
        if (!(s_hat > 0.0)) s_hat = theta[1];
    }
    else {
        a_hat = theta[0];
        s_hat = theta[1];
    }

    // Эмпирика (К–М)
    EmpiricalKM emp = kaplan_meier_Itype(S.x, S.r);
    vector<double> X_emp = emp.x_sorted, Y_emp;
    Y_emp.reserve(emp.F_emp.size());
    for (double F : emp.F_emp) {
        double Fclip = min(max(F, 1e-12), 1.0 - 1e-12); //чтобы не улетало в бесконечность
        Y_emp.push_back(5.0 + norm_ppf(Fclip)); //вычисление квантиля
    }

    // уровни p (считали при чтении)
    vector<double> P;
    {
        ifstream pin("Out/" + TAG + ".probs");
        for (string line; getline(pin, line);) {
            if (line.empty()) continue;
            for (char& c : line) if (c == ',') c = '.';
            try {
                double p = stod(line);
                P.push_back(p);
            }
            catch (...) {
                // пропускаем мусорные строки
            }
        }
    }

    // ковариация оценок и интервалы квантилей
    auto V = cov_normal_asymp(S.x, S.r, a_hat, s_hat);
    const double beta = 0.95;
    const double z_beta = norm_ppf((1.0 + beta) / 2.0); //двусторонний доверительный интервал

    vector<double> Xlow, Xest, Xupp, Yline;
    Xlow.reserve(P.size()); Xest.reserve(P.size()); Xupp.reserve(P.size()); Yline.reserve(P.size());

    for (double p_raw : P) {
        const double p = min(max(p_raw, 1e-12), 1.0 - 1e-12);
        const double z = norm_ppf(p);
        const double xp = a_hat + z * s_hat;

        const double var_xp = V[0][0] + 2.0 * V[0][1] * z + V[1][1] * z * z;
        const double sd_xp = sqrt(std::max(0.0, var_xp / double(S.n)));

        Xest.push_back(xp);
        Xlow.push_back(xp - z_beta * sd_xp);
        Xupp.push_back(xp + z_beta * sd_xp);
        Yline.push_back(5.0 + z);
    }

    const int nc = 4;
    vector<int> m = { (int)X_emp.size(), (int)P.size(), (int)P.size(), (int)P.size() };

    ofstream xo("Out/" + TAG + ".xout");
    if (!xo) { cerr << "[ERROR] Не удалось открыть Out/" << TAG << ".xout\n"; return 1; }
    xo << nc << "\n";
    for (int i = 0; i < nc; i++) xo << m[i] << (i + 1 < nc ? " " : "");
    xo << "\n";
    xo << setprecision(17);
    for (double v : X_emp) xo << v << " "; xo << "\n";
    for (double v : Xlow) xo << v << " "; xo << "\n";
    for (double v : Xest) xo << v << " "; xo << "\n";
    for (double v : Xupp) xo << v << " "; xo << "\n";
    for (double v : Y_emp) xo << v << " "; xo << "\n";
    for (double v : Yline) xo << v << " "; xo << "\n";
    for (double v : Yline) xo << v << " "; xo << "\n";
    for (double v : Yline) xo << v << " "; xo << "\n";
    xo.close();

    ofstream out("Out/" + TAG + ".out");
    out << "Method:" << TAG << "\n";
    out << "n=" << S.n << "\n";
    out << "X\n"; for (double v : S.x) out << v << " , "; out << "\n";
    out << "R\n"; for (int v : S.r) out << v << " , "; out << "\n";
    out << fixed << setprecision(12);
    out << "a_hat=" << a_hat << "\n";
    out << "sigma_hat=" << s_hat << "\n";
    out << "Cov[a,s]:\n";
    out << V[0][0] << " " << V[0][1] << "\n";
    out << V[1][0] << " " << V[1][1] << "\n";
    out << "P\n"; for (double p : P) out << p << " ; "; out << "\n";
    out << "Xp_low\n"; for (double v : Xlow) out << v << " ; "; out << "\n";
    out << "Xp\n"; for (double v : Xest) out << v << " ; "; out << "\n";
    out << "Xp_up\n"; for (double v : Xupp) out << v << " ; "; out << "\n";
    out.close();
    cout << "Выполнено: " + TAG;

    return 0;
}



//================================Вейбулл===========================

int run_weibull() {
    try {
        const string TAG = "MLE_Weibull";
        auto S = read_input_weibull(TAG);

        // Оценки ММП
        auto est = weibull_mle_2par(S.x, S.r);
        double c_hat = est.first;
        double b_hat = est.second;

        // Защита от NaN/Inf и плохих значений
        if (!(isfinite(c_hat) && isfinite(b_hat)) || c_hat <= 0.0 || b_hat <= 0.0) {
            // фолбэк на линейную регрессию на "бумаге Вейбулла"
            auto est_fb = weibull_regression_fallback(S.x, S.r);
            c_hat = est_fb.first;
            b_hat = est_fb.second;
        }
        // дополнительная "страховка" от диких чисел
        if (!(isfinite(c_hat) && isfinite(b_hat)) || c_hat <= 0.0 || b_hat <= 0.0) {
            throw runtime_error("Не удалось получить корректные оценки Вейбулла (b,c)");
        }


        // Эмпирика (К–М)
        auto emp = kaplan_meier_Itype(S.x, S.r);
        vector<double> X_emp = emp.x_sorted, Y_emp; Y_emp.reserve(emp.F_emp.size());
        for (double F : emp.F_emp) {
            double Fclip = min(max(F, 1e-12), 1.0 - 1e-12); //чтобы не брать логарифм нуля
            double y = 5.0 + log(log(1.0 / (1.0 - Fclip)));
            Y_emp.push_back(y);
        }

        // Уровни p
        vector<double> P;
        {
            ifstream pin("Out/" + TAG + ".probs");
            for (string line; getline(pin, line);) if (!line.empty()) {
                for (char& c : line) if (c == ',') c = '.';
                try { P.push_back(stod(line)); }
                catch (...) {}
            }
        }

        const double beta = 0.95; // или прочитай из файла, см. пункт C ниже
        const double z_beta = norm_ppf((1.0 + beta) / 2.0);

        // Ковариация и эффективный размер (число r=0)
        auto V_eff = cov_weibull_asymp_eff(S.x, S.r, c_hat, b_hat);
        auto& Vpq = V_eff.first;
        const int n_eff = std::max(1, V_eff.second);

        std::vector<double> Xlow, Xest, Xupp, Yline;
        Xlow.reserve(P.size()); Xest.reserve(P.size()); Xupp.reserve(P.size()); Yline.reserve(P.size());

        for (double p_raw : P) {
            const double pclip = std::min(std::max(p_raw, 1.0e-12), 1.0 - 1.0e-12);
            const double t = -std::log(1.0 - pclip);
            const double xp = c_hat * std::pow(t, 1.0 / b_hat);

            // Градиент в (p=ln c, q=1/b):
            const double g0 = xp;
            const double g1 = xp * std::log(t);

            double var_xp = Vpq[0][0] * g0 * g0
                + 2.0 * Vpq[0][1] * g0 * g1
                + Vpq[1][1] * g1 * g1;

            // Масштабируем дисперсию на размер выборки
            var_xp = std::max(0.0, var_xp / static_cast<double>(n_eff));
            const double sd_xp = std::sqrt(var_xp); 

            Xest.push_back(xp);
            Xlow.push_back(xp - z_beta * sd_xp);
            Xupp.push_back(xp + z_beta * sd_xp);

            const double y = 5.0 + std::log(std::log(1.0 / (1.0 - pclip)));
            Yline.push_back(y);
        }


        const int nc = 4;
        vector<int> m = { (int)X_emp.size(), (int)P.size(), (int)P.size(), (int)P.size() };

        ofstream xo("Out/" + TAG + ".xout");
        xo << nc << "\n";
        for (int i = 0; i < nc; i++) xo << m[i] << (i + 1 < nc ? " " : "");
        xo << "\n";
        xo << setprecision(17);
        for (double v : X_emp) xo << v << " "; xo << "\n";
        for (double v : Xlow) xo << v << " "; xo << "\n";
        for (double v : Xest) xo << v << " "; xo << "\n";
        for (double v : Xupp) xo << v << " "; xo << "\n";
        for (double v : Y_emp) xo << v << " "; xo << "\n";
        for (double v : Yline) xo << v << " "; xo << "\n";
        for (double v : Yline) xo << v << " "; xo << "\n";
        for (double v : Yline) xo << v << " "; xo << "\n";
        xo.close();

        ofstream out("Out/" + TAG + ".out");
        out << "Method:" << TAG << "\n";
        out << "n=" << S.n << "\n";
        out << "X\n"; for (double v : S.x) out << v << " , "; out << "\n";
        out << "R\n"; for (int v : S.r) out << v << " , "; out << "\n";
        out << fixed << setprecision(12);
        out << "c_hat=" << c_hat << "\n";
        out << "b_hat=" << b_hat << "\n";
        out << "P\n"; for (double p : P) out << p << " ; "; out << "\n";
        out << "Xp_low\n"; for (double v : Xlow) out << v << " ; "; out << "\n";
        out << "Xp\n"; for (double v : Xest) out << v << " ; "; out << "\n";
        out << "Xp_up\n"; for (double v : Xupp) out << v << " ; "; out << "\n";
        out.close();
        cout << "Выполнено: " + TAG;
        return 0;
    }
    catch (const exception& e) {
        cerr << "[ERROR] " << e.what() << endl;
        return 1;
    }
}
