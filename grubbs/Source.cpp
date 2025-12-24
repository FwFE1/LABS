#include "Source.h"


double normPPF(double p) {
    if (p <= 0 || p >= 1) return 0;
    boost::math::normal_distribution<>d(0, 1);
    return(quantile(d, p));
}

void insertSus(vector<double>& sample, double alpha) {
    cout << "Вставка подозрительного значения: " << endl;
    int susIndex = rand() % (sample.size() - 1);
    double susNumber = alpha * 1000 + 5;
    cout << "x[" << susIndex << "] = " << sample[susIndex] << " ---- > ";
    sample[susIndex] = susNumber;
    cout << "x[" << susIndex << "] = " << sample[susIndex] << endl;
}

void showVector(vector <double>& sample) {
    for (int i = 0; i < sample.size(); i++) {
        cout << "x[" << i << "] = " << sample[i] << endl;
    }
}

double studentPPF(double alpha, const int freedomDegree) {
    students_t dist(freedomDegree);
    double t_crit = quantile(complement(dist, alpha));
    return t_crit;
}

double maxVector(vector <double>& vect) {
    double mx = INT_MIN;
    for (int i = 0; i < vect.size(); i++) {
        if (vect[i] >= mx) mx = vect[i];
    }
    return mx;
}

double minVector(vector <double>& vect) {
    double mn = INT_MAX;
    for (int i = 0; i < vect.size(); i++) {
        if (vect[i] <= mn) mn = vect[i];
    }
    return mn;
}

void generateSample(vector <double>& sample, double& alpha, int& state, double& a, double& s) {
    const int n = 20 + rand() % 21;
    a = -5. + 10. * rand() / static_cast<double>(RAND_MAX);
    s = 0.5 + 2.5 * rand() / static_cast<double>(RAND_MAX);
    sample.resize(n);
    for (int i = 0; i < n; i++) {
        double z = rand() / static_cast<double>(RAND_MAX);
        double znorm = normPPF(z);
        double y = znorm * s + a;
        sample[i] = y;
    }
    alpha = 0.05;
    state = 0;
}

double calculateMean(vector <double>& sample) {
    double mean = 0;
    int n = sample.size();
    for (int i = 0; i < n; i++) {
        mean += sample[i];
    }
    mean /= n;
    cout << "Среднее = " << mean << endl;
    return mean;
}

double calculateStandartDeviation(vector <double>& sample, double mean) {
    double standartDeviation = 0;
    int n = sample.size();
    for (int i = 0; i < n; i++) {
        standartDeviation += (sample[i] - mean) * (sample[i] - mean);
    }
    standartDeviation /= n - 1;
    standartDeviation = sqrt(standartDeviation);
    cout << "СКО = " << standartDeviation << endl;
    return standartDeviation;
}

double calculateGrubbsStat(vector <double>& sample, double mean, double standartDeviation, int state) {
    double result = 0;
    double x_1 = maxVector(sample);
    double x_n = minVector(sample);
    double u_1 = (x_1 - mean) / standartDeviation;
    double u_n = (mean - x_n) / standartDeviation;
    if (state == 0) result = max(u_1, u_n); // макс и мин
    else if (state == 1) result = u_1;// макс
    else result = u_n;  // мин
    cout << "u = " << result << endl;
    return result;
}

double calculateUAlpha(const int sampleSize, double alpha, bool state) {
    int n = sampleSize;
    double t = studentPPF(alpha / (2 * n), n - 2);
    if (state) t = studentPPF(alpha / n, n - 2);
    double u_alpha = (n - 1) * (sqrt((t * t) / (n * (n - 2 + t * t))));
    cout << "u_alpha = " << u_alpha << endl;
    return u_alpha;
}



bool performGrubbsTest(vector <double>& sample, double alpha, int state, double a, double s, string FileName) {
    double mean = calculateMean(sample);
    double standartDeviation = calculateStandartDeviation(sample, mean);
    double u = calculateGrubbsStat(sample, mean, standartDeviation, state);
    double u_alpha = calculateUAlpha(sample.size(), alpha, state);
    bool result = (u <= u_alpha);
    writeGrubbsOutput(FileName, sample, alpha, state, a, s, mean, standartDeviation, u, u_alpha, result);
    return result; // true - нулевая гипотеза выполняется, false - не выполняется

}


void readStartConfiguration(const string& fileName,
    vector<double>& sample,
    double& alpha,
    int& state,
    double& a,
    double& s) {
    string inputPath = "Inp/" + fileName;
    ifstream fin(inputPath);

    bool needGenerate = false;

    if (!fin.is_open()) {
        cout << "Файл " << inputPath << " не найден. Будет сгенерирована новая выборка." << endl;
        needGenerate = true;
    }
    else {
        // проверяем, пуст ли файл
        fin.seekg(0, ios::end);
        if (fin.tellg() == 0) {
            cout << "Файл " << inputPath << " пуст. Будет сгенерирована новая выборка." << endl;
            needGenerate = true;
        }
        else {
            fin.seekg(0, ios::beg);
            int n;
            if (!(fin >> n)) {
                cout << "Не удалось прочитать размер выборки. Будет сгенерирована новая выборка." << endl;
                needGenerate = true;
            }
            else {
                if (!(fin >> state)) {
                    cout << "Не удалось прочитать state. Будет сгенерирована новая выборка." << endl;
                    needGenerate = true;
                }
                else if (!(fin >> alpha)) {
                    cout << "Не удалось прочитать alpha. Будет сгенерирована новая выборка." << endl;
                    needGenerate = true;
                }
                else if (!(fin >> a)) {
                    cout << "Не удалось прочитать a. Будет сгенерирована новая выборка." << endl;
                    needGenerate = true;
                }
                else if (!(fin >> s)) {
                    cout << "Не удалось прочитать s. Будет сгенерирована новая выборка." << endl;
                    needGenerate = true;
                }
                else {
                    sample.resize(n);
                    bool readOk = true;
                    for (int i = 0; i < n; i++) {
                        if (!(fin >> sample[i])) {
                            cout << "Ошибка чтения x[" << i << "]. Будет сгенерирована новая выборка." << endl;
                            readOk = false;
                            break;
                        }
                    }
                    if (!readOk) {
                        needGenerate = true;
                    }
                }
            }
        }
        fin.close();
    }

    if (needGenerate) {
        // генерируем новую выборку N(a, s)
        generateSample(sample, alpha, state, a, s);

        // один раз вставляем подозрительное значение в сгенерированную выборку
        insertSus(sample, alpha);

        ofstream fout(inputPath);
        if (!fout.is_open()) {
            cout << "Не удалось создать файл " << inputPath << ". Данные будут использоваться только в памяти." << endl;
        }
        else {
            fout.setf(ios::fixed);
            fout.precision(6);

            // Формат файла:
            // n
            // state
            // alpha
            // a
            // s
            // x[0]
            // ...
            fout << sample.size() << endl;
            fout << state << endl;
            fout << alpha << endl;
            fout << a << endl;
            fout << s << endl;

            for (size_t i = 0; i < sample.size(); i++) {
                fout << sample[i] << endl;
            }

            fout.close();

            cout << "Создан файл " << inputPath << " со сгенерированными входными данными (с одним подозрительным значением)." << endl;
        }
    }

    cout << "Входные данные:" << endl;
    cout << "n     = " << sample.size() << endl;
    cout << "state = " << state << endl;
    cout << "alpha = " << alpha << endl;
    cout << "a     = " << a << endl;
    cout << "s     = " << s << endl;
}


void writeGrubbsOutput(const string& fileName,
    const vector<double>& sample,
    double alpha,
    int state,
    double a,
    double s,
    double mean,
    double standartDeviation,
    double u,
    double u_alpha,
    bool h0Accepted) {
    string outputPath = "Out/" + fileName;
    ofstream fout(outputPath);

    if (!fout.is_open()) {
        cout << "Не удалось открыть файл " << outputPath << " для записи." << endl;
        return;
    }

    fout.setf(ios::fixed);
    fout.precision(6);

    fout << "Критерий Граббса для нормального распределения" << endl;
    fout << "---------------------------------------------" << endl;
    fout << "Размер выборки n      = " << sample.size() << endl;
    fout << "state                 = " << state << endl;
    fout << "Уровень значимости α  = " << alpha << endl;
    fout << endl;

    fout << "Истинные параметры распределения (использовались при генерации):" << endl;
    fout << "a (мат. ожидание)     = " << a << endl;
    fout << "s (СКО)               = " << s << endl;
    fout << endl;

    fout << "Выборка:" << endl;
    for (size_t i = 0; i < sample.size(); i++) {
        fout << "x[" << i << "] = " << sample[i] << endl;
    }
    fout << endl;

    fout << "Оценки по выборке:" << endl;
    fout << "Среднее значение           = " << mean << endl;
    fout << "Стандартное отклонение     = " << standartDeviation << endl;
    fout << "Наблюдаемая статистика u   = " << u << endl;
    fout << "Критическое значение u_alpha   = " << u_alpha << endl;
    fout << endl;

    fout << "Вывод: ";
    if (h0Accepted) {
        fout << "u <= u_α, нулевая гипотеза НЕ отвергается (подозрительных выбросов нет)." << endl;
    }
    else {
        fout << "u > u_α, нулевая гипотеза отвергается (в выборке есть подозрительный выброс)." << endl;
    }

    fout.close();

    cout << "Результаты записаны в файл " << outputPath << endl;
}







