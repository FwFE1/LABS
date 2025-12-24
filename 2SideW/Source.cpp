#include "Source.h"

#include <numeric>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <stdexcept>

#include <boost/math/distributions/normal.hpp>

Sample::Sample()
    : sample(), mean(0.0), stdDev(0.0) {
}

Sample::Sample(const std::vector<double>& sample)
    : sample(sample) {
    this->mean = calculateMean();
    this->stdDev = calculateStandardDeviation();
}

double Sample::calculateMean() const {
    if (sample.empty()) {
        return 0.0;
    }
    double sum = std::accumulate(sample.begin(), sample.end(), 0.0);
    return sum / static_cast<double>(sample.size());
}

double Sample::calculateStandardDeviation() const {
    const int n = static_cast<int>(sample.size());
    if (n <= 1) {
        return 0.0;
    }

    double variance = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = sample[static_cast<std::size_t>(i)] - mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(n - 1);
    return std::sqrt(variance);
}

double Sample::getMean() const {
    return mean;
}

double Sample::getStdDev() const {
    return stdDev;
}

int Sample::getSampleSize() const {
    return static_cast<int>(sample.size());
}

const std::vector<double>& Sample::getSample() const {
    return sample;
}

double normPPF(double alpha) {
    if (alpha <= 0 || alpha >= 1) return 0;
    boost::math::normal_distribution<>d(0, 1);
    return(quantile(d, alpha));
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

Sample generateSampleNormal(double mean, double stdDev, int n, bool shift) {
    if (stdDev <= 0.0) {
        stdDev = 0.1;
    }

    std::mt19937& gen = globalGenerator();
    if (n <= 0) {
        n = generateIntNumberFromAToB(gen, 20, 40);
    }

    double delta = shift ? 0.05 : 0.0;

    std::vector<double> sample(static_cast<std::size_t>(n));
    std::normal_distribution<double> dist(mean + delta, stdDev);
    for (double& el : sample) {
        el = dist(gen);
    }

    return Sample(sample);
}


double calculateWilcoxonStat(const Sample& sample1, const Sample& sample2)
{
    const std::vector<double>& x = sample1.getSample();
    const std::vector<double>& y = sample2.getSample();

    int m = static_cast<int>(x.size());
    int n = static_cast<int>(y.size());

    if (m <= 0 || n <= 0) {
        throw std::invalid_argument("Выборки для критерия Уилкоксона не должны быть пустыми");
    }

    // Объединяем две выборки в (значение, метка_группы)
    std::vector<std::pair<double, int>> united;
    united.reserve(static_cast<std::size_t>(m + n));

    for (int i = 0; i < m; ++i) {
        united.emplace_back(x[static_cast<std::size_t>(i)], 0); // группа 0
    }
    for (int j = 0; j < n; ++j) {
        united.emplace_back(y[static_cast<std::size_t>(j)], 1); // группа 1
    }

    // Сортируем по значениям
    std::sort(united.begin(), united.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first < b.first;
        });

    // Назначаем ранги 1..N без учёта связей (ties)
    const int N = m + n;
    std::vector<int> ranks(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        ranks[static_cast<std::size_t>(i)] = i + 1;
    }

    // Берём сумму рангов для меньшей по объёму выборки
    int smallGroupId = (m <= n) ? 0 : 1;
    double W = 0.0;

    for (int i = 0; i < N; ++i) {
        if (united[static_cast<std::size_t>(i)].second == smallGroupId) {
            W += static_cast<double>(ranks[static_cast<std::size_t>(i)]);
        }
    }

    return W;
}


std::pair<double, double> calculateApproximately(int m, int n, double alpha, bool oneSided)
{
    std::pair<double, double> wCrit{ 0.0, 0.0 };

    double mu = (static_cast<double>(m) * (m + n + 1)) / 2.0;
    double sigma = std::sqrt(static_cast<double>(m) * n * (m + n + 1) / 12.0);

    if (oneSided) {
        // Односторонний: берём границу по одному хвосту
        double z_alpha = normPPF(alpha); // z_{alpha}, как в N(0,1)
        wCrit.first = mu + z_alpha * sigma;
        wCrit.second = 0.0;
    }
    else {
        // Двусторонний: нижняя граница по alpha/2, верхняя по 1 - alpha/2
        double z_low = normPPF(alpha / 2.0);
        double z_high = normPPF(1.0 - alpha / 2.0);

        wCrit.first = mu + z_low * sigma; // нижняя граница
        wCrit.second = mu + z_high * sigma; // верхняя граница
    }

    return wCrit;
}


// ------------------------------------------------------------
// Биномиальный коэффициент C(n, k) в виде long double
// ------------------------------------------------------------
long double binomial_coefficient(int n, int k) {
    if (k < 0 || k > n) {
        return 0.0L;
    }
    if (k == 0 || k == n) {
        return 1.0L;
    }
    if (k > n - k) {
        k = n - k;
    }

    long double result = 1.0L;
    for (int i = 1; i <= k; ++i) {
        result *= static_cast<long double>(n - k + i);
        result /= static_cast<long double>(i);
    }
    return result;
}

// ------------------------------------------------------------
// Точное распределение суммы рангов W через DP.
// На выходе:
//   minW, maxW, maxU,
//   pU[u] = P(U = u), u = 0..maxU,
//   cdfU[u] = P(U <= u).
// ------------------------------------------------------------
void buildExactWilcoxonDistribution(
    int m,
    int n,
    int& minW,
    int& maxW,
    int& maxU,
    std::vector<long double>& pU,
    std::vector<long double>& cdfU
) {
    if (m <= 0 || n <= 0) {
        throw std::invalid_argument("m и n должны быть положительными");
    }

    const int N = m + n;

    // минимальная и максимальная сумма рангов W
    minW = m * (m + 1) / 2;
    maxW = m * (2 * N - m + 1) / 2; // сумма m наибольших рангов

    maxU = m * n;

    // DP-таблица: dp[i][s] = количество способов выбрать i рангов с суммой s
    std::vector<std::vector<long double>> dp(
        m + 1,
        std::vector<long double>(maxW + 1, 0.0L)
    );

    dp[0][0] = 1.0L;

    // Для каждого ранга r от 1 до N обновляем dp
    for (int r = 1; r <= N; ++r) {
        int max_i = std::min(r, m);
        for (int i = max_i; i >= 1; --i) {
            for (int s = maxW; s >= r; --s) {
                if (dp[i - 1][s - r] > 0.0L) {
                    dp[i][s] += dp[i - 1][s - r];
                }
            }
        }
    }

    // Общее число способов выбрать m рангов из N
    long double total_combinations = binomial_coefficient(N, m);

    // Распределение по U: U = W - minW, U = 0..m*n
    pU.assign(maxU + 1, 0.0L);

    for (int W = minW; W <= maxW; ++W) {
        int U = W - minW;
        if (U >= 0 && U <= maxU) {
            long double countWays = dp[m][W];
            if (countWays > 0.0L) {
                pU[U] = countWays / total_combinations;
            }
        }
    }

    // Накопленные вероятности CDF: F(u) = P(U <= u)
    cdfU.assign(maxU + 1, 0.0L);
    long double cumulative = 0.0L;
    for (int u = 0; u <= maxU; ++u) {
        cumulative += pU[u];
        cdfU[u] = cumulative;
    }
}

// ------------------------------------------------------------
// Нахождение квантиля U_alpha: минимальный u, такой что F(u) >= alpha
// ------------------------------------------------------------
int findQuantileU(
    const std::vector<long double>& cdfU,
    int maxU,
    long double alpha
) {
    if (alpha <= 0.0L) {
        return 0;
    }
    if (alpha >= 1.0L) {
        return maxU;
    }

    for (int u = 0; u <= maxU; ++u) {
        if (cdfU[u] >= alpha) {
            return u;
        }
    }
    return maxU;
}

// ------------------------------------------------------------
// Перевод квантиля U -> квантиль W
// ------------------------------------------------------------
int convertUtoW(int minW, int U_quantile) {
    return minW + U_quantile;
}


std::pair<double, double> calculatePrecisely(int m, int n, double alpha, bool oneSided)
{
    std::pair<double, double> wCrit{ 0.0, 0.0 };

    int minW = 0;
    int maxW = 0;
    int maxU = 0;
    std::vector<long double> pU;
    std::vector<long double> cdfU;

    buildExactWilcoxonDistribution(m, n, minW, maxW, maxU, pU, cdfU);

    if (oneSided) {
        // Односторонний: левая граница по уровню alpha
        long double alphaLeft = static_cast<long double>(alpha);
        int U_alpha = findQuantileU(cdfU, maxU, alphaLeft);
        int W_alpha = convertUtoW(minW, U_alpha);

        wCrit.first = static_cast<double>(W_alpha);
        wCrit.second = 0.0;
    }
    else {
        // Двусторонний: хвосты по alpha/2
        long double tail = static_cast<long double>(alpha) / 2.0L;

        // Левая граница: минимальный u, такой что F(u) >= alpha/2
        int U_left = findQuantileU(cdfU, maxU, tail);
        int W_left = convertUtoW(minW, U_left);

        // Правая граница через симметрию U ↔ maxU - U
        int U_right = maxU - U_left;
        int W_right = convertUtoW(minW, U_right);

        wCrit.first = static_cast<double>(W_left);
        wCrit.second = static_cast<double>(W_right);
    }

    return wCrit;
}


std::pair<double, double> calculateWCrit(
    const Sample& sample1,
    const Sample& sample2,
    double alpha,
    bool oneSided
)
{
    int m1 = sample1.getSampleSize();
    int n1 = sample2.getSampleSize();

    if (m1 <= 0 || n1 <= 0) {
        throw std::invalid_argument("Размеры выборок для критерия Уилкоксона должны быть положительными");
    }

    // m – размер выборки, для которой считается сумма рангов W (меньшая выборка)
    // n – размер второй (большей) выборки
    int m = (m1 <= n1) ? m1 : n1;
    int n = (m1 <= n1) ? n1 : m1;

    std::pair<double, double> wCrit{ 0.0, 0.0 };

    if (m + n <= 40) {
        // Точное распределение
        wCrit = calculatePrecisely(m, n, alpha, oneSided);
    }
    else {
        // Нормальное приближение
        wCrit = calculateApproximately(m, n, alpha, oneSided);
    }

    return wCrit;
}

WilcoxonDecision wilcoxonTwoSidedDecision(
    const Sample& sample1,
    const Sample& sample2,
    double alpha
)
{
    // 1. Считаем наблюдаемую статистику W
    double W_obs = calculateWilcoxonStat(sample1, sample2);

    // 2. Получаем критические значения W_lower и W_upper
    //    false  -> двусторонний критерий
    std::pair<double, double> wCrit = calculateWCrit(sample1, sample2, alpha, false);

    double W_lower = wCrit.first;
    double W_upper = wCrit.second;

    // 3. Правило:
    //    если W_obs <= W_lower или W_obs >= W_upper  -> отвергаем H0
    //    иначе                                         -> не отвергаем H0
    if (W_obs <= W_lower || W_obs >= W_upper) {
        return WilcoxonDecision::RejectH0;
    }
    else {
        return WilcoxonDecision::AcceptH0;
    }
}


void generateTwoSidedWilcoxonInp(double alpha)
{
    std::mt19937& gen = globalGenerator();

    // Размеры выборок: от 8 до 20 элементов
    int m = generateIntNumberFromAToB(gen, 8, 30);
    int n = generateIntNumberFromAToB(gen, 8, 30);

    // Случайно решаем, будет ли сдвиг второй выборки
    int shiftFlagInt = generateIntNumberFromAToB(gen, 0, 1);
    bool shiftSecond = (shiftFlagInt != 0);

    // Первая выборка: N(0, 1), без сдвига
    Sample sample1 = generateSampleNormal(0.0, 1.0, m, false);

    // Вторая выборка: N(0, 1) со сдвигом 0.05, если shiftSecond == true
    Sample sample2 = generateSampleNormal(0.0, 1.0, n, shiftSecond);

    std::ofstream fout("Inp/twosidedwilcoxon.inp");
    if (!fout.is_open()) {
        throw std::runtime_error("Не удалось открыть файл Inp/twosidedwilcoxon.inp для записи");
    }

    fout.setf(std::ios::fixed);
    fout.precision(6);

    // Строка 1: alpha
    fout << alpha << '\n';

    // Строка 2: m n
    fout << m << ' ' << n << '\n';

    // Строка 3: shiftSecond (0 или 1)
    fout << (shiftSecond ? 1 : 0) << '\n';

    // Строка 4: выборка 1
    const std::vector<double>& x = sample1.getSample();
    for (int i = 0; i < m; ++i) {
        fout << x[static_cast<std::size_t>(i)];
        if (i + 1 < m) {
            fout << ' ';
        }
    }
    fout << '\n';

    // Строка 5: выборка 2
    const std::vector<double>& y = sample2.getSample();
    for (int j = 0; j < n; ++j) {
        fout << y[static_cast<std::size_t>(j)];
        if (j + 1 < n) {
            fout << ' ';
        }
    }
    fout << '\n';
}

void runTwoSidedWilcoxonFromFile()
{
    // ---------- Чтение входных данных ----------

    std::ifstream fin("Inp/twosidedwilcoxon.inp");
    if (!fin.is_open()) {
        throw std::runtime_error("Не удалось открыть файл Inp/twosidedwilcoxon.inp для чтения");
    }

    double alpha = 0.0;
    int m_inp = 0;
    int n_inp = 0;
    int shiftFlagInt = 0;

    if (!(fin >> alpha)) {
        throw std::runtime_error("Ошибка чтения alpha из Inp/twosidedwilcoxon.inp");
    }
    if (!(fin >> m_inp >> n_inp)) {
        throw std::runtime_error("Ошибка чтения размеров выборок из Inp/twosidedwilcoxon.inp");
    }
    if (!(fin >> shiftFlagInt)) {
        throw std::runtime_error("Ошибка чтения флага сдвига из Inp/twosidedwilcoxon.inp");
    }

    bool shiftSecond = (shiftFlagInt != 0);

    if (m_inp <= 0 || n_inp <= 0) {
        throw std::runtime_error("Размеры выборок из файла Inp/twosidedwilcoxon.inp должны быть положительными");
    }

    std::vector<double> x(static_cast<std::size_t>(m_inp));
    std::vector<double> y(static_cast<std::size_t>(n_inp));

    for (int i = 0; i < m_inp; ++i) {
        if (!(fin >> x[static_cast<std::size_t>(i)])) {
            throw std::runtime_error("Ошибка чтения элементов первой выборки из Inp/twosidedwilcoxon.inp");
        }
    }

    for (int j = 0; j < n_inp; ++j) {
        if (!(fin >> y[static_cast<std::size_t>(j)])) {
            throw std::runtime_error("Ошибка чтения элементов второй выборки из Inp/twosidedwilcoxon.inp");
        }
    }

    fin.close();

    Sample sample1(x);
    Sample sample2(y);

    int m1 = sample1.getSampleSize();
    int n1 = sample2.getSampleSize();
    int N = m1 + n1;

    int m_small = (m1 <= n1) ? m1 : n1;
    int n_large = (m1 <= n1) ? n1 : m1;
    int smallGroupId = (m1 <= n1) ? 0 : 1;

    bool useExact = (m_small + n_large <= 40);

    // ---------- Статистика критерия ----------

    double W_obs = calculateWilcoxonStat(sample1, sample2);
    std::pair<double, double> wCrit = calculateWCrit(sample1, sample2, alpha, false);

    double W_lower = wCrit.first;
    double W_upper = wCrit.second;

    // ---------- Подготовка рангов для подробного вывода ----------

    const std::vector<double>& xs = sample1.getSample();
    const std::vector<double>& ys = sample2.getSample();

    std::vector<std::pair<double, int>> united;
    united.reserve(static_cast<std::size_t>(N));

    for (int i = 0; i < m1; ++i) {
        united.emplace_back(xs[static_cast<std::size_t>(i)], 0); // выборка 1
    }
    for (int j = 0; j < n1; ++j) {
        united.emplace_back(ys[static_cast<std::size_t>(j)], 1); // выборка 2
    }

    std::sort(
        united.begin(),
        united.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first < b.first;
        }
    );

    std::vector<int> ranks(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        ranks[static_cast<std::size_t>(i)] = i + 1;
    }

    // ---------- Запись подробного отчёта ----------

    std::ofstream fout("Out/twosidedwilcoxon.out");
    if (!fout.is_open()) {
        throw std::runtime_error("Не удалось открыть файл Out/twosidedwilcoxon.out для записи");
    }

    fout.setf(std::ios::fixed);
    fout.precision(6);

    fout << "================ Двусторонний критерий Уилкоксона (Mann–Whitney) ================\n\n";

    fout << "Параметры моделирования\n";
    fout << "-----------------------\n";
    fout << "Уровень значимости alpha = " << alpha << "\n";
    fout << "Размер первой выборки m1 = " << m1 << "\n";
    fout << "Размер второй выборки n1 = " << n1 << "\n";
    fout << "Общий объём N = m1 + n1 = " << N << "\n";
    fout << "Меньшая выборка по объёму: выборка " << (smallGroupId + 1)
        << " (m = " << m_small << ")\n\n";

    fout << "Генерация данных\n";
    fout << "----------------\n";
    fout << "Обе выборки сгенерированы из нормального распределения N(0, 1).\n";
    fout << "Для второй выборки применён небольшой сдвиг среднего на +0.05: "
        << (shiftSecond ? "ДА" : "НЕТ") << "\n\n";

    fout << "Данные выборок\n";
    fout << "--------------\n";

    fout << "Выборка 1 (m1 = " << m1 << "):\n";
    for (int i = 0; i < m1; ++i) {
        fout << "x[" << (i + 1) << "] = " << xs[static_cast<std::size_t>(i)] << "\n";
    }
    fout << "\n";

    fout << "Выборка 2 (n1 = " << n1 << "):\n";
    for (int j = 0; j < n1; ++j) {
        fout << "y[" << (j + 1) << "] = " << ys[static_cast<std::size_t>(j)] << "\n";
    }
    fout << "\n";

    fout << "Описательная статистика\n";
    fout << "------------------------\n";
    fout << "Выборка 1: выборочное среднее = " << sample1.getMean()
        << ", выборочное СКО = " << sample1.getStdDev() << "\n";
    fout << "Выборка 2: выборочное среднее = " << sample2.getMean()
        << ", выборочное СКО = " << sample2.getStdDev() << "\n\n";

    fout << "Ранжирование объединённой выборки\n";
    fout << "---------------------------------\n";
    fout << "Столбцы: индекс в отсортированной объединённой выборке, значение, номер выборки (1 или 2), ранг\n";

    for (int i = 0; i < N; ++i) {
        fout << (i + 1) << "  "
            << united[static_cast<std::size_t>(i)].first << "  "
            << (united[static_cast<std::size_t>(i)].second + 1) << "  "
            << ranks[static_cast<std::size_t>(i)] << "\n";
    }
    fout << "\n";

    fout << "Статистика критерия\n";
    fout << "--------------------\n";
    fout << "Наблюдаемое значение статистики W (сумма рангов меньшей выборки):\n";
    fout << "W_obs = " << W_obs << "\n\n";

    fout << "Критические значения (двусторонний критерий)\n";
    fout << "--------------------------------------------\n";
    fout << "Режим вычисления критических значений: "
        << (useExact ? "ТОЧНОЕ распределение (динамическое программирование)"
            : "НОРМАЛЬНОЕ ПРИБЛИЖЕНИЕ") << "\n";

    fout << "Нижняя критическая граница W_lower = " << W_lower << "\n";
    fout << "Верхняя критическая граница W_upper = " << W_upper << "\n";

    if (!useExact) {
        double mu = (static_cast<double>(m_small) * (m_small + n_large + 1)) / 2.0;
        double sigma = std::sqrt(
            static_cast<double>(m_small) * n_large * (m_small + n_large + 1) / 12.0
        );

        fout << "Теоретическое математическое ожидание mu = " << mu << "\n";
        fout << "Теоретическое стандартное отклонение sigma = " << sigma << "\n";
    }
    fout << "\n";

    fout << "Решение\n";
    fout << "-------\n";
    fout << "H0: распределения двух выборок совпадают (нет сдвига по медиане).\n";
    fout << "H1: распределения различаются (медианы или положение распределений отличаются).\n";

    bool reject =
        (W_obs <= W_lower) ||
        (W_obs >= W_upper);

    if (reject) {
        fout << "Так как W_obs попадает в критическую область (W_obs <= W_lower или W_obs >= W_upper),\n";
        fout << "нулевая гипотеза H0 ОТВЕРГАЕТСЯ при уровне значимости alpha = " << alpha << ".\n";
    }
    else {
        fout << "Так как W_obs находится внутри интервала [W_lower; W_upper],\n";
        fout << "нет оснований отвергать нулевую гипотезу H0 при уровне значимости alpha = " << alpha << ".\n";
    }

    fout << "\n";

    fout.close();
}













