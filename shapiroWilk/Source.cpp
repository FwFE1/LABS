#define _USE_MATH_DEFINES

#include "Source.h"

#include <numeric>
#include <cmath>
#include <fstream>
#include <limits>


#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/beta.hpp>

// Реализация методов класса Sample

Sample::Sample(const std::vector<double>& sample)
    : sample(sample) {
    this->mean = calculateMean();
    this->stdDev = calculateStandardDeviation();
}

double Sample::calculateMean() const {
    double sum = std::accumulate(this->sample.begin(),
        this->sample.end(),
        0.0);
    const int n = static_cast<int>(this->sample.size());
    if (n == 0) {
        return 0.0;
    }
    return sum / static_cast<double>(n);
}

double Sample::calculateStandardDeviation() const {
    const int n = static_cast<int>(this->sample.size());
    if (n <= 1) {
        return 0.0;
    }

    double standartDeviation = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = this->sample[i] - this->mean;
        standartDeviation += diff * diff;
    }
    standartDeviation /= static_cast<double>(n - 1);
    standartDeviation = std::sqrt(standartDeviation);
    return standartDeviation;
}

double Sample::getMean() const {
    return this->mean;
}

double Sample::getStdDev() const {
    return this->stdDev;
}

int Sample::getSampleSize() const {
    return static_cast<int>(this->sample.size());
}

const std::vector<double>& Sample::getSample() const {
    return this->sample;
}


std::mt19937& globalGenerator() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

int generateIntNumberFromAToB(std::mt19937& generator, int a, int b) {
    std::uniform_int_distribution<int> uni(a, b);
    return uni(generator);
}

Sample generateSample(double mean, double stdDev) {
    // проверка на корректность СКО
    if (stdDev <= 0.0) {
        stdDev = 0.1;
    }

    std::mt19937& gen = globalGenerator();

    // генерация размера выборки
    const int n = generateIntNumberFromAToB(gen);

    std::vector<double> sample(n);

    // генерация выборки
    std::normal_distribution<double> dist(mean, stdDev);
    for (double& el : sample) {
        el = dist(gen);
    }
    std::sort(sample.begin(), sample.end()); // приведение к вариационному ряду
    return Sample(sample);
}

double calculateSquaredDeviations(const Sample& sample) {
    double res = 0;
    for (double el : sample.getSample()) {
        double diff = el - sample.getMean();
        res += diff*diff;
    }
    return res;
}


// ===============================================================================

// Нормальная CDF N(0,1)
double normalCDF(double x) {
    static const boost::math::normal_distribution<double> nd(0.0, 1.0);
    return boost::math::cdf(nd, x);
}

// Квантиль N(0,1)
double normalQuantile01(double p) {
    if (p <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    if (p >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    static const boost::math::normal_distribution<double> nd(0.0, 1.0);
    return boost::math::quantile(nd, p);
}

// Плотность N(0,1)
double normalPDF01(double x) {
    static const boost::math::normal_distribution<double> nd(0.0, 1.0);
    return boost::math::pdf(nd, x);
}

std::vector<double> calculateNormalOrderStatisticsExpectations(int n) {
    if (n <= 0) {
        throw std::invalid_argument("calculateNormalOrderStatisticsExpectations: n must be > 0");
    }

    std::vector<double> expectations(n);

    for (int i = 1; i <= n; ++i) {
        // p строго внутри (0,1), поэтому проблем с quantile быть не должно
        double p = static_cast<double>(i) / (n + 1.0);
        double u_p = normalQuantile01(p);
        double f_u = normalPDF01(u_p);

        // подстраховка на случай очень маленькой плотности
        if (f_u <= 0.0) {
            expectations[i - 1] = u_p;
            continue;
        }

        double f_prime = -u_p * f_u;

        // основной член
        double e = u_p;

        // первая поправка Дэйвида
        double corr1 = (p * (1.0 - p)) /
            (2.0 * (n + 2.0)) *
            (f_prime / (f_u * f_u));

        e += corr1;

        expectations[i - 1] = e;
    }

    return expectations;
}

std::vector<std::vector<double>> calculateOrderStatisticsCovariance(int n) {
    if (n <= 0) {
        throw std::invalid_argument("calculateOrderStatisticsCovariance: n must be > 0");
    }

    std::vector<std::vector<double>> cov(n, std::vector<double>(n, 0.0));

    // Сразу считаем p_i, u_i, f_i, чтобы не дёргать Boost в каждой итерации
    std::vector<double> p(n);
    std::vector<double> u(n);
    std::vector<double> f(n);

    for (int i = 0; i < n; ++i) {
        p[i] = static_cast<double>(i + 1) / (n + 1.0);
        u[i] = normalQuantile01(p[i]);
        f[i] = normalPDF01(u[i]);

        if (f[i] <= 0.0) {
            // подстраховка: чтобы не делить на 0, зададим очень маленькое значение
            f[i] = 1e-16;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {

            if (i == j) {
                cov[i][j] = p[i] * (1.0 - p[i]) /
                    ((n + 2.0) * f[i] * f[i]);
            }
            else {
                double p_min = (p[i] < p[j]) ? p[i] : p[j];
                double p_max = (p[i] > p[j]) ? p[i] : p[j];

                cov[i][j] = (p_min * (1.0 - p_max)) /
                    ((n + 2.0) * f[i] * f[j]);
            }
        }
    }

    return cov;
}

// ----------------------------------------------------------------------------
// Решение системы V * y = b методом Гаусса–Жордана (без выбора ведущего элемента)
// V и b передаём по значению (копии), чтобы не портить исходные данные.
// ----------------------------------------------------------------------------
std::vector<double> solveLinearSystem(std::vector<std::vector<double>> V,
    std::vector<double> b) {
    int n = static_cast<int>(V.size());
    if (n == 0) {
        throw std::invalid_argument("solveLinearSystem: matrix size is zero");
    }
    if (static_cast<int>(b.size()) != n) {
        throw std::invalid_argument("solveLinearSystem: size mismatch");
    }

    for (int i = 0; i < n; ++i) {
        double pivot = V[i][i];
        if (std::fabs(pivot) < 1e-14) {
            throw std::runtime_error("solveLinearSystem: pivot is too small");
        }

        // Нормируем i-ю строку
        for (int j = i; j < n; ++j) {
            V[i][j] /= pivot;
        }
        b[i] /= pivot;

        // Обнуляем столбец i во всех остальных строках
        for (int k = 0; k < n; ++k) {
            if (k == i) continue;
            double factor = V[k][i];
            for (int j = i; j < n; ++j) {
                V[k][j] -= factor * V[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    return b; // теперь вектор b — решение y
}


std::vector<double> computeAi_viaIntegrals(int n) {
    if (n <= 0) {
        throw std::invalid_argument("computeAi_viaIntegrals: n must be > 0");
    }

    // 1. Вектор математических ожиданий нормальных порядковых статистик
    std::vector<double> alpha = calculateNormalOrderStatisticsExpectations(n);

    // 2. Ковариационная матрица V
    std::vector<std::vector<double>> V = calculateOrderStatisticsCovariance(n);

    // 3. Решаем V * y = alpha
    std::vector<double> y = solveLinearSystem(V, alpha);

    // 4. Нормируем: a_i = y_i / ||y||
    double norm2 = 0.0;
    for (double val : y) {
        norm2 += val * val;
    }
    if (norm2 <= 0.0) {
        throw std::runtime_error("computeAi_viaIntegrals: non-positive norm");
    }

    double norm = std::sqrt(norm2);
    std::vector<double> a(n);
    for (int i = 0; i < n; ++i) {
        a[i] = y[i] / norm;
    }

    return a;
}

// ===============================================================================





double calculateBCoef(const std::vector<double>& a_i, const Sample& sample) {
    double result = 0;
    for (std::size_t i = 0; i < sample.getSampleSize(); i++) {
        result += a_i[i] * sample.getSample()[i];
    }
    return result;
}

double calculateWilkStat(const double b, const double sSquared) {
    return (b * b) / sSquared;
}

double calculateWAlpha(int n) {
    // Табличные значения W_crit для уровня значимости alpha = 0.05
    // по Приложению П1 (n = 3..50).
    // Индекс массива = n, элементы 0,1,2 не используются.
    static const double Wcrit05[51] = {
        0.0,   // 0 - нет теста
        0.0,   // 1 - нет теста
        0.0,   // 2 - нет теста
        0.767, // 3
        0.748, // 4
        0.762, // 5
        0.788, // 6
        0.803, // 7
        0.818, // 8
        0.829, // 9
        0.842, // 10
        0.850, // 11
        0.859, // 12
        0.866, // 13
        0.874, // 14
        0.881, // 15
        0.887, // 16
        0.892, // 17
        0.897, // 18
        0.901, // 19
        0.905, // 20
        0.908, // 21
        0.911, // 22
        0.914, // 23
        0.916, // 24
        0.918, // 25
        0.920, // 26
        0.923, // 27
        0.924, // 28
        0.926, // 29
        0.927, // 30
        0.929, // 31
        0.930, // 32
        0.931, // 33
        0.933, // 34
        0.934, // 35
        0.935, // 36
        0.936, // 37
        0.938, // 38
        0.939, // 39
        0.940, // 40
        0.941, // 41
        0.942, // 42
        0.943, // 43
        0.944, // 44
        0.945, // 45
        0.945, // 46
        0.946, // 47
        0.947, // 48
        0.947, // 49
        0.947  // 50
    };

    if (n < 3 || n > 50) {
        throw std::invalid_argument(
            "Критические значения W по Агамирову заданы только для 3 <= n <= 50");
    }

    return Wcrit05[n];
}

Sample generateInputFileWithNewSample(const std::string& filePath,
    double mean,
    double stdDev) {

    Sample sample = generateSample(mean, stdDev);

    std::ofstream fout(filePath, std::ios::trunc);
    if (!fout.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для записи: " + filePath);
    }

    const int n = sample.getSampleSize();
    const std::vector<double>& data = sample.getSample();

    fout << n << '\n';
    for (int i = 0; i < n; ++i) {
        fout << data[i];
        if (i + 1 < n) {
            fout << ' ';
        }
    }
    fout << '\n';

    return sample;
}

void writeShapiroWilkResultToFile(const std::string& filePath,
    const Sample& sample,
    const std::vector<double>& a_i,
    double alpha) {

    std::ofstream fout(filePath, std::ios::trunc);
    if (!fout.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для записи: " + filePath);
    }

    const int n = sample.getSampleSize();
    const double mean = sample.getMean();
    const double stdDev = sample.getStdDev();

    double sSquared = calculateSquaredDeviations(sample);
    double b = calculateBCoef(a_i, sample);
    double W_observed = calculateWilkStat(b, sSquared);
    double W_crit = calculateWAlpha(n);

    fout << "Критерий Шапиро-Уилка\n";
    fout << "Уровень значимости alpha = " << alpha << "\n\n";

    fout << "Размер выборки n = " << n << "\n";
    fout << "Выборочное среднее x̄ = " << mean << "\n";
    fout << "Выборочное СКО s = " << stdDev << "\n\n";

    fout << "Сумма квадратов отклонений s^2 = " << sSquared << "\n";
    fout << "Коэффициент b = " << b << "\n";
    fout << "Наблюдаемое значение статистики Wнабл = " << W_observed << "\n";
    fout << "Критическое значение Wкр = " << W_crit << "\n\n";

    if (W_observed >= W_crit) {
        fout << "Вывод: Wнабл > Wкр, нет оснований отвергать нулевую гипотезу.\n";
        fout << "Распределение можно считать нормальным.\n";
    }
    else {
        fout << "Вывод: Wнабл < Wкр, нулевая гипотеза отвергается.\n";
        fout << "Распределение нельзя считать нормальным.\n";
    }
}
