#pragma once

#include <vector>
#include <utility>
#include <random>
#include <string>

// Класс Sample: хранит выборку и её характеристики
class Sample {
private:
    std::vector<double> sample;
    double mean = 0.0;
    double stdDev = 0.0;

public:
    // Конструктор из готового вектора значений
    Sample(const std::vector<double>& sample);

    // Вычисляет среднее арифметическое выборки
    double calculateMean() const;

    // Вычисляет исправленное стандартное отклонение (делитель n - 1)
    double calculateStandardDeviation() const;

    double getMean() const;
    double getStdDev() const;
    int getSampleSize() const;
    const std::vector<double>& getSample() const;
};

// Глобальный генератор случайных чисел mt19937
std::mt19937& globalGenerator();

// Генерация целого числа в диапазоне [a, b]
int generateIntNumberFromAToB(std::mt19937& generator, int a = 20, int b = 45);

// Генерация выборки из нормального распределения N(mean, stdDev)
std::vector<double> generateSample(double mean, double stdDev);

// Генерация нескольких выборок (групп)
std::vector<Sample> generateListOfSamples(int numberOfSamples);

// Общее среднее по всем выборкам
double calculateGeneralMean(const std::vector<Sample>& samples);

// Межгрупповая дисперсия
double calculateOuterDispersion(const std::vector<Sample>& samples,
    double generalMean);

// Внутригрупповая дисперсия
double calculateInnerDispersion(const std::vector<Sample>& samples);

// F-статистика Фишера
double calculateFisherStat(double sOut, double sIn);

// Степени свободы (f1, f2)
std::pair<int, int> calculateFreedomDegreees(const std::vector<Sample>& samples);

// Критическое значение F_alpha
double calculateFAlpha(const std::pair<int, int>& freedomDegrees, double alpha);

// Подсчитать средние значения в параллельных измерениях samples.
// Формат ввода:
// Количество серий: m — количество измерений;
// Далее m строк: n_i x_{i1} x_{i2} ... x_{i n_i}
void generateInputFileIfEmpty(const std::string& fileName, int numberOfSamples, std::vector<Sample>& samples);

// Провести дисперсионный анализ ANOVA для групп данных.
// Формат вывода для дисперсионного анализа.
void writeAnovaToFile(const std::string& fileName, double alpha, const std::vector<Sample>& samples);



