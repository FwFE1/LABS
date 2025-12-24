#pragma once

#include <vector>
#include <utility>
#include <random>
#include <string>
#include <functional>
#include <stdexcept>

// Класс Sample: хранит выборку, её среднее и СКО
class Sample {
private:
    std::vector<double> sample;
    double mean = 0.0;
    double stdDev = 0.0;

public:
    // Конструктор от константной ссылки на вектор
    Sample(const std::vector<double>& sample);

    // Расчет среднего значения выборки
    double calculateMean() const;

    // Расчет выборочного стандартного отклонения (деление на n - 1)
    double calculateStandardDeviation() const;

    double getMean() const;
    double getStdDev() const;
    int getSampleSize() const;
    const std::vector<double>& getSample() const;
};


// Генератор mt19937 (глобальный)
std::mt19937& globalGenerator();

// Целое число в диапазоне [a, b]
int generateIntNumberFromAToB(std::mt19937& generator, int a = 20, int b = 45);

// Генерация одной нормальной выборки N(mean, stdDev)
Sample generateSample(double mean, double stdDev);


// ============================= ии слоп 💣💣💣💣💣💣==================================================
double integrateSimpson(const std::function<double(double)>& f,
    double a,
    double b,
    int steps = 100);
double normalQuantile01(double u);
double alpha_nr(int n, int r);
double V_rr(int n, int r, double alpha_nr_value);
double innerIntegral_for_Vrs(double v, int r, int s);
double V_rs(int n, int r, int s,
    double alpha_r, double alpha_s);
std::vector<double> solveLinearSystem(std::vector<std::vector<double>> V,
    std::vector<double> b);
std::vector<double> computeAi_viaIntegrals(int n);
// ============================= конец ии слопа 💣💣💣💣💣💣==================================================

double calculateBCoef(const std::vector<double>& a_i, const Sample& sample);
double calculateWilkStat(double b, double s);
double calculateWAlpha(int n);

// Генерация новой выборки, запись её в входной файл и возврат Sample.
// filePath — полный путь к файлу, включая имя (например, "Inp/shapiro.in").
// mean, stdDev — параметры нормального распределения N(mean, stdDev).
Sample generateInputFileWithNewSample(const std::string& filePath, double mean, double stdDev);

// Запись результатов критерия Шапиро–Уилка в выходной файл.
// filePath — полный путь к файлу (например, "Out/shapiro.out").
// sample — выборка, по которой считается критерий.
// a_i — коэффициенты a_i для Шапиро–Уилка.
// alpha — уровень значимости.
void writeShapiroWilkResultToFile(const std::string& filePath, const Sample& sample, const std::vector<double>& a_i, double alpha);

