#include "Source.h"

#include <numeric>
#include <cmath>
#include <fstream>

#include "boost/math/distributions/fisher_f.hpp"

// Реализация класса Sample

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


// Реализация функций для ANOVA анализа

std::mt19937& globalGenerator() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

int generateIntNumberFromAToB(std::mt19937& generator, int a, int b) {
    std::uniform_int_distribution<int> uni(a, b);
    return uni(generator);
}

std::vector<double> generateSample(double mean, double stdDev) {
    // Защита от некорректного СКО
    if (stdDev <= 0.0) {
        stdDev = 0.1;
    }

    std::mt19937& gen = globalGenerator();

    /// Генерируем размер выборки
    const int n = generateIntNumberFromAToB(gen);

    std::vector<double> sample(n);

    /// Генерируем значения
    std::normal_distribution<double> dist(mean, stdDev);
    for (double& el : sample) {
        el = dist(gen);
    }

    return sample;
}

std::vector<Sample> generateListOfSamples(const int numberOfSamples) {
    std::vector<Sample> result;
    result.reserve(numberOfSamples);

    std::vector<double> means(numberOfSamples, 1.0);
    std::vector<double> stdDevs(numberOfSamples, 0.3);

    for (int i = 0; i < numberOfSamples; ++i) {
        result.emplace_back(generateSample(means[i], stdDevs[i]));
    }

    return result;
}

double calculateGeneralMean(const std::vector<Sample>& samples) {
    double numerator = 0.0;
    int totalN = 0;

    for (const Sample& sample : samples) {
        numerator += sample.getMean() * sample.getSampleSize();
        totalN += sample.getSampleSize();
    }

    if (totalN == 0) {
        return 0.0;
    }

    return numerator / static_cast<double>(totalN);
}

double calculateOuterDispersion(const std::vector<Sample>& samples, double generalMean) {
    const int k = static_cast<int>(samples.size());
    if (k <= 1) {
        return 0.0;
    }

    const int f_1 = k - 1;

    double numerator = 0.0;
    for (const Sample& el : samples) {
        double diff = el.getMean() - generalMean;
        numerator += el.getSampleSize() * diff * diff;
    }

    return numerator / static_cast<double>(f_1);
}

double calculateInnerDispersion(const std::vector<Sample>& samples) {
    int totalN = 0;
    for (const Sample& el : samples) {
        totalN += el.getSampleSize();
    }

    const int k = static_cast<int>(samples.size());
    const int f_2 = totalN - k;

    if (f_2 <= 0) {
        return 0.0;
    }

    double numerator = 0.0;
    for (const Sample& el : samples) {
        const int n_i = el.getSampleSize();
        const double s = el.getStdDev();
        numerator += (n_i - 1) * s * s;
    }

    return numerator / static_cast<double>(f_2);
}

double calculateFisherStat(double sOut, double sIn) {
    if (sIn == 0.0) {
        return 0.0; //  или std::numeric_limits<double>::infinity()
    }

    double F = sOut / sIn;
    return F;
}

std::pair<int, int> calculateFreedomDegreees(const std::vector<Sample>& samples) {
    std::pair<int, int> result;

    const int k = static_cast<int>(samples.size());
    int totalN = 0;

    for (const Sample& el : samples) {
        totalN += el.getSampleSize();
    }

    int f_1 = k - 1;
    int f_2 = totalN - k;

    result.first = f_1;
    result.second = f_2;

    return result;
}

double calculateFAlpha(const std::pair<int, int>& freedomDegrees, double alpha) {
    boost::math::fisher_f_distribution<double> dist(freedomDegrees.first, freedomDegrees.second);
    double F_alpha = quantile(boost::math::complement(dist, alpha));
    return F_alpha;
}

void generateInputFileIfEmpty(const std::string& fileName, int numberOfSamples, std::vector<Sample>& samples) {
    /// Генерируем выборки
    samples = generateListOfSamples(numberOfSamples);

    // Записываем в файл
    std::ofstream fout(fileName, std::ios::trunc);
    if (!fout.is_open()) {
        return;
    }

    fout << samples.size() << '\n';

    for (const Sample& s : samples) {
        const std::vector<double>& v = s.getSample();
        fout << v.size();
        for (double x : v) {
            fout << ' ' << std::setprecision(10) << x;
        }
        fout << '\n';
    }
}


void writeAnovaToFile(const std::string& fileName, double alpha, const std::vector<Sample>& samples) {
    if (samples.empty()) {
        return;
    }

    double generalMean = calculateGeneralMean(samples);
    double sOut = calculateOuterDispersion(samples, generalMean);
    double sIn = calculateInnerDispersion(samples);
    double F = calculateFisherStat(sOut, sIn);
    std::pair<int, int> df = calculateFreedomDegreees(samples);
    double F_alpha = calculateFAlpha(df, alpha);

    std::ofstream fout(fileName, std::ios::trunc);
    if (!fout.is_open()) {
        return;
    }

    fout << std::setprecision(10);

    fout << "Дисперсионный анализ (ANOVA)\n\n";

    fout << "Уровень значимости alpha = " << alpha << '\n';
    fout << "Число групп (выборок) k =" << samples.size() << '\n';
    fout << "Общее среднее X.. = " << generalMean << "\n\n";

    for (std::size_t i = 0; i < samples.size(); ++i) {
        fout << "Выборка " << (i + 1)
            << ": n = " << samples[i].getSampleSize()
            << ", mean = " << samples[i].getMean()
            << ", stdDev = " << samples[i].getStdDev()
            << '\n';
    }

    fout << "\nМежгрупповая дисперсия S_out = " << sOut << '\n';
    fout << "Внутригрупповая дисперсия S_in = " << sIn << '\n';
    fout << "Наблюдаемое значение F = " << F << '\n';
    fout << "Степени свободы: f1 = " << df.first
        << ", f2 = " << df.second << '\n';
    fout << "Критическое значение F_(1-alpha) = " << F_alpha << "\n\n";

    fout << "Вывод: ";
    if (F <= F_alpha) {
        fout << "H0 принимается (средние значения групп статистически неразличимы).\n";
    }
    else {
        fout << "H0 отвергается (средние значения групп статистически различны).\n";
    }
}
