#include "Source.h"


double trueMean1 = 0.0;
double trueMean2 = 0.0;
double trueStd1 = 0.0;
double trueStd2 = 0.0;

double normPPF(double p) {
    if (p <= 0.0 || p >= 1.0) {
        return 0.0;
    }
    boost::math::normal_distribution<> d(0.0, 1.0);
    double x = quantile(d, p);
    return x;
}

double studentPPF(double alpha, double freedomDegree) {
    students_t dist(freedomDegree);
    double t_crit = quantile(complement(dist, alpha));
    return t_crit;
}

double calculateMean(vector<double>& sample) {
    double mean = 0.0;
    int n = static_cast<int>(sample.size());
    for (int i = 0; i < n; ++i) {
        mean += sample[i];
    }
    if (n > 0) {
        mean /= static_cast<double>(n);
    }
    cout << "Среднее = " << mean << endl;
    return mean;
}

double calculateStandartDeviation(vector<double>& sample, double mean) {
    double standartDeviation = 0.0;
    int n = static_cast<int>(sample.size());
    for (int i = 0; i < n; ++i) {
        double diff = sample[i] - mean;
        standartDeviation += diff * diff;
    }
    if (n > 1) {
        standartDeviation /= static_cast<double>(n - 1);
        standartDeviation = sqrt(standartDeviation);
    }
    cout << "СКО = " << standartDeviation << endl;
    return standartDeviation;
}

void generateSamples(vector<double>& sample1, vector<double>& sample2, double& alpha) {
    // Первая выборка
    const int n1 = 20 + rand() % 21;
    double a1 = 1.00;  // генеральное мат. ожидание первой совокупности
    double s1 = 0.10;  // генеральное СКО первой совокупности

    sample1.resize(n1);
    for (int i = 0; i < n1; ++i) {
        double z = rand() / static_cast<double>(RAND_MAX);
        double znorm = normPPF(z);
        double y = znorm * s1 + a1;
        sample1[i] = y;
    }

    // Вторая выборка
    const int n2 = 20 + rand() % 21;
    double a2 = 1.00;  // генеральное мат. ожидание второй совокупности
    double s2 = 0.15;  // генеральное СКО второй совокупности

    sample2.resize(n2);
    for (int i = 0; i < n2; ++i) {
        double z = rand() / static_cast<double>(RAND_MAX);
        double znorm = normPPF(z);
        double y = znorm * s2 + a2;
        sample2[i] = y;
    }

    alpha = 0.05;

    // Сохраняем генеральные параметры, чтобы потом вывести в файл
    trueMean1 = a1;
    trueStd1 = s1;
    trueMean2 = a2;
    trueStd2 = s2;
}

double calculateFisherStat(double s1, double s2) {
    double F = max(s1 * s1, s2 * s2) / min(s1 * s1, s2 * s2);
    return F;
}

vector<double> calculateFreedomDegrees(vector<double>& sample1, double s1, vector<double>& sample2, double s2) {
    int n1 = static_cast<int>(sample1.size());
    int n2 = static_cast<int>(sample2.size());
    vector<double> freedomDegrees(2);
    if (s1 > s2) {
        freedomDegrees[0] = static_cast<double>(n1 - 1);
        freedomDegrees[1] = static_cast<double>(n2 - 1);
    }
    else {
        freedomDegrees[1] = static_cast<double>(n1 - 1);
        freedomDegrees[0] = static_cast<double>(n2 - 1);
    }
    return freedomDegrees;
}

double calculateFAlpha(vector<double> freedomDegrees, double alpha) {
    boost::math::fisher_f_distribution<double> dist(freedomDegrees[0], freedomDegrees[1]);
    double F_alpha = quantile(complement(dist, alpha));
    return F_alpha;
}

bool performFisherTest(vector<double>& sample1,
    vector<double>& sample2,
    double& alpha,
    vector<double>& stdDvtns,
    vector<double>& freedomDegrees,
    vector<int>& samplesSize,
    vector<double>& means) {
    double mean1 = calculateMean(sample1);
    double s1 = calculateStandartDeviation(sample1, mean1);
    double mean2 = calculateMean(sample2);
    double s2 = calculateStandartDeviation(sample2, mean2);

    stdDvtns[0] = s1;
    stdDvtns[1] = s2;
    means[0] = mean1;
    means[1] = mean2;
    samplesSize[0] = static_cast<int>(sample1.size());
    samplesSize[1] = static_cast<int>(sample2.size());

    double F = calculateFisherStat(s1, s2);
    freedomDegrees = calculateFreedomDegrees(sample1, s1, sample2, s2);
    double F_alpha = calculateFAlpha(freedomDegrees, alpha);

    bool result = (F <= F_alpha);
    return result; // true - дисперсии равны, false - не равны
}

double calculateStudentStatEqualDisp(vector<double>& stdDvtns,
    vector<double>& freedomDegrees,
    vector<int>& samplesSize,
    vector<double>& means) {
    double numeratorS = freedomDegrees[0] * (stdDvtns[0] * stdDvtns[0]) +
        freedomDegrees[1] * (stdDvtns[1] * stdDvtns[1]);
    double denumeratorS = freedomDegrees[0] + freedomDegrees[1];
    double s = sqrt(numeratorS / denumeratorS);

    double numeratorT = means[0] - means[1];
    double denumeratorT = s * sqrt(1.0 / static_cast<double>(samplesSize[0]) +
        1.0 / static_cast<double>(samplesSize[1]));
    double t = numeratorT / denumeratorT;
    return t;
}

double calculateStudentStatNotEqualDisp(vector<double>& stdDvtns,
    vector<double>& freedomDegrees,
    vector<int>& samplesSize,
    vector<double>& means) {
    double numeratorC = (stdDvtns[0] * stdDvtns[0]) / static_cast<double>(samplesSize[0]);
    double denumeratorC = (stdDvtns[0] * stdDvtns[0]) / static_cast<double>(samplesSize[0]) +
        (stdDvtns[1] * stdDvtns[1]) / static_cast<double>(samplesSize[1]);
    double c = numeratorC / denumeratorC;

    double numeratorF = 1.0;
    double denumeratorF = (c * c) / freedomDegrees[0] +
        (1.0 - c) * (1.0 - c) / freedomDegrees[1];
    double f = numeratorF / denumeratorF;

    freedomDegrees.resize(3);
    freedomDegrees[2] = f;

    double numeratorT = means[0] - means[1];
    double denumeratorT = sqrt((stdDvtns[0] * stdDvtns[0]) / static_cast<double>(samplesSize[0]) +
        (stdDvtns[1] * stdDvtns[1]) / static_cast<double>(samplesSize[1]));
    double t = numeratorT / denumeratorT;
    return t;
}

bool performStudentTest(bool fisherTestResult,
    double alpha,
    vector<double>& stdDvtns,
    vector<int>& samplesSize,
    vector<double>& means) {
    vector<double> freedomDegrees = {
        static_cast<double>(samplesSize[0]) - 1.0,
        static_cast<double>(samplesSize[1]) - 1.0
    };

    double t = 0.0;

    if (fisherTestResult) {
        t = calculateStudentStatEqualDisp(stdDvtns, freedomDegrees, samplesSize, means);
        double t_alpha_halved = studentPPF(alpha / 2.0, freedomDegrees[0] + freedomDegrees[1]);
        bool result = (fabs(t) <= t_alpha_halved);
        return result; // true --> a1 = a2. false --> a1 != a2
    }
    else {
        t = calculateStudentStatNotEqualDisp(stdDvtns, freedomDegrees, samplesSize, means);
        double t_alpha_halved = studentPPF(alpha / 2.0, freedomDegrees[2]);
        bool result = (fabs(t) <= t_alpha_halved);
        return result; // true --> a1 = a2. false --> a1 != a2
    }
}

// Эта версия КАЖДЫЙ раз при запуске генерирует новые выборки
// и перезаписывает входной файл Inp/studentfisher.inp
void readStartConfiguration(vector<double>& sample1, vector<double>& sample2, double& alpha) {
    const string inputFileName = "Inp/studentfisher.inp";

    // Всегда генерируем новые выборки
    generateSamples(sample1, sample2, alpha);

    // Перезаписываем входной файл при каждом запуске
    ofstream fout(inputFileName);
    if (!fout.is_open()) {
        cout << "Не удалось открыть файл " << inputFileName << "для записи сгенерированных данных." << endl;
        return;
    }

    fout << fixed << setprecision(6);

    // Записываем alpha
    fout << alpha << '\n';

    // Первая выборка
    fout << sample1.size() << '\n';
    for (int i = 0; i < static_cast<int>(sample1.size()); ++i) {
        fout << sample1[i];
        if (i + 1 < static_cast<int>(sample1.size())) {
            fout << ' ';
        }
    }
    fout << '\n';

    // Вторая выборка
    fout << sample2.size() << '\n';
    for (int i = 0; i < static_cast<int>(sample2.size()); ++i) {
        fout << sample2[i];
        if (i + 1 < static_cast<int>(sample2.size())) {
            fout << ' ';
        }
    }
    fout << '\n';

    cout << "Сгенерированные данные записаны в " << inputFileName << endl;
}

// Формат выходных данных в файл fileName (например, Out/studentfisher.out):
// Добавлено: вывод генеральных мат. ожиданий и генеральных дисперсий
void writeOutput(const string& fileName,
    double alpha,
    const vector<double>& sample1,
    const vector<double>& sample2,
    const vector<double>& stdDvtns,
    const vector<double>& freedomDegrees,
    const vector<int>& samplesSize,
    const vector<double>& means,
    bool fisherResult,
    bool studentResult) {
    ofstream fout(fileName, ios::trunc);
    if (!fout.is_open()) {
        cout << "Не удалось открыть файл " << fileName << " для записи результатов." << endl;
        return;
    }

    fout << fixed << setprecision(6);

    fout << "Критерий Фишера и критерий Стьюдента для двух выборок\n\n";
    fout << "Уровень значимости alpha = " << alpha << "\n\n";
    fout << "Первая выборка (n1 = ";
    if (!samplesSize.empty()) {
        fout << samplesSize[0];
    }
    else {
        fout << sample1.size();
    }
    fout << "):\n";

    for (int i = 0; i < static_cast<int>(sample1.size()); ++i) {
        fout << sample1[i];
        if (i + 1 < static_cast<int>(sample1.size())) {
            fout << ' ';
        }
    }
    fout << "\n";

    if (means.size() >= 1) {
        fout << "Выборочное среднее m1 = " << means[0] << "\n";
    }
    if (stdDvtns.size() >= 1) {
        fout << "Выборочное СКО s1 = " << stdDeviations[0] << "\n";
    }

    // Генеральные параметры для первой выборки
    fout << "Генеральное мат. ожидание mu1 = " << trueMean1 << "\n";
    fout << "Генеральное СКО sigma1 = " << trueStd1 << "\n";

    fout << "\n";

    fout << "Вторая выборка (n2 = ";
    if (samplesSize.size() >= 2) {
        fout << samplesSize[1];
    }
    else {
        fout << sample2.size();
    }
    fout << "):\n";

    for (int i = 0; i < static_cast<int>(sample2.size()); ++i) {
        fout << sample2[i];
        if (i + 1 < static_cast<int>(sample2.size())) {
            fout << ' ';
        }
    }
    fout << "\n";

    if (means.size() >= 2) {
        fout << "Выборочное среднее m2 = " << means[1] << "\n";
    }
    if (stdDvtns.size() >= 2) {
        fout << "Выборочное СКО s2 = " << stdDeviations[1] << "\n";
    }

    // Генеральные параметры для второй выборки
    fout << "Генеральное мат. ожидание mu2 = " << trueMean2 << "\n";
    fout << "Генеральное СКО sigma2 = " << trueStd2 << "\n";
    fout << "\n";

    if (!freedomDegrees.empty()) {
        fout << "Степени свободы: nu1 = " << freedomDegrees[0];
        if (freedomDegrees.size() >= 2) {
            fout << ", nu2 = " << freedomDegrees[1];
        }
        if (freedomDegrees.size() >= 3) {
            fout << ", nu* = " << freedomDegrees[2];
        }
        fout << "\n\n";
    }

    fout << "Результат критерия Фишера (проверка равенства дисперсий): ";
    if (fisherResult) {
        fout << "дисперсии можно считать равными.\n";
    }
    else {
        fout << "дисперсии различаются.\n";
    }

    fout << "Результат критерия Стьюдента (проверка равенства средних): ";
    if (studentResult) {
        fout << "математические ожидания можно считать равными.\n";
    }
    else {
        fout << "математические ожидания различаются.\n";
    }

    fout.close();
}
