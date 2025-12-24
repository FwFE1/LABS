#pragma once

#include <vector>
#include <random>
#include <string>
#include <utility>

// --------------------------- Класс выборки ---------------------------

class Sample {
private:
    std::vector<double> sample;
    double mean;
    double stdDev;

     // Вычисление выборочного среднего по текущей выборке
    double calculateMean() const;

    // Вычисление выборочного СКО по текущей выборке
    double calculateStandardDeviation() const;

public:
    // Пустая выборка (нуль элементов, среднее и СКО равны 0)
    Sample();

    // Инициализация выборкой из вектора
    explicit Sample(const std::vector<double>& sample);

    // Получить выборочное среднее
    double getMean() const;

    // Получить выборочное СКО
    double getStdDev() const;

    // Получить объём выборки
    int getSampleSize() const;

     // Получить константную ссылку на вектор выборки
    const std::vector<double>& getSample() const;
};

// --------------------------- Общий функционал ---------------------------

// Глобальный генератор случайных чисел mt19937
std::mt19937& globalGenerator();

// Случайное целое число в диапазоне [a, b]
int generateIntNumberFromAToB(std::mt19937& generator, int a, int b);

// Обратная функция распределения N(0,1): квантиль нормального распределения
double normPPF(double alpha);

// Генерация нормальной выборки:
// mean   – математическое ожидание,
// stdDev – СКО (если stdDev <= 0, берётся небольшое положительное значение),
// n      – объём выборки (если n <= 0, генерируется случайный размер),
// shift  – если true, к среднему добавляется небольшой сдвиг (0.05).
Sample generateSampleNormal(double mean, double stdDev, int n, bool shift = false);

// --------------------- Критерий Уилкоксона (двусторонний) ---------------------

// Статистика W: сумма рангов меньшей по размеру выборки
double calculateWilcoxonStat(const Sample& sample1, const Sample& sample2);

// Приближённые критические значения W по нормальному распределению
// oneSided = true  -> односторонний критерий
// oneSided = false -> двусторонний критерий (возвращает нижнюю и верхнюю границы)
std::pair<double, double> calculateApproximately(int m, int n, double alpha, bool oneSided);

// Биномиальный коэффициент C(n, k) в виде long double
long double binomial_coefficient(int n, int k);

// Построение точного распределения суммы рангов W через DP.
// На выходе:
//   minW, maxW, maxU,
//   pU[u]   = P(U = u), u = 0..maxU,
//   cdfU[u] = P(U <= u), u = 0..maxU.
void buildExactWilcoxonDistribution(
    int m,
    int n,
    int& minW,
    int& maxW,
    int& maxU,
    std::vector<long double>& pU,
    std::vector<long double>& cdfU
);

// Нахождение квантиля U_alpha: минимальный u, такой что F(u) >= alpha
int findQuantileU(
    const std::vector<long double>& cdfU,
    int maxU,
    long double alpha
);

// Перевод квантиля U -> квантиль W
int convertUtoW(int minW, int U_quantile);

// Точные критические значения W (для малых выборок)
// oneSided = true  -> односторонний критерий
// oneSided = false -> двусторонний критерий (нижняя и верхняя границы)
std::pair<double, double> calculatePrecisely(int m, int n, double alpha, bool oneSided);

// Объединяющая функция: выбор точного или приближённого метода для критических значений W
std::pair<double, double> calculateWCrit(
    const Sample& sample1,
    const Sample& sample2,
    double alpha,
    bool oneSided
);

// Решение по критерию Уилкоксона
enum class WilcoxonDecision
{
    AcceptH0,  // Критерий не отверг H0 (различия НЕзначимы)
    RejectH0   // Критерий отверг H0 (различия значимы)
};

// Двусторонний критерий Уилкоксона: вернуть решение (принять/отвергнуть H0)
WilcoxonDecision wilcoxonTwoSidedDecision(
    const Sample& sample1,
    const Sample& sample2,
    double alpha
);

// --------------------- Работа с файлами Inp/Out ---------------------

// Генерация новой конфигурации задачи для двустороннего критерия Уилкоксона.
// Каждый вызов создаёт новый файл Inp/twosidedwilcoxon.inp.
void generateTwoSidedWilcoxonInp(double alpha);

// Чтение задачи из Inp/twosidedwilcoxon.inp, выполнение критерия
// и вывод максимально подробного отчёта в Out/twosidedwilcoxon.out.
void runTwoSidedWilcoxonFromFile();
