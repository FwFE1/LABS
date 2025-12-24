#ifndef SOURCE_H
#define SOURCE_H

#include <vector>
#include <random>
#include <climits>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/fisher_f.hpp>

using namespace std;
using namespace boost::math;

// Генеральные параметры распределений, используемых при генерации выборок
extern double trueMean1;
extern double trueMean2;
extern double trueStd1;   // сигма (СКО), не дисперсия
extern double trueStd2;   // сигма (СКО), не дисперсия

double normPPF(double p);
double studentPPF(double alpha, double freedomDegree);
double calculateMean(vector<double>& sample);
double calculateStandartDeviation(vector<double>& sample, double mean);
void generateSamples(vector<double>& sample1, vector<double>& sample2, double& alpha);
double calculateFisherStat(double s1, double s2);
vector<double> calculateFreedomDegrees(vector<double>& sample1, double s1, vector<double>& sample2, double s2);
double calculateFAlpha(vector<double> freedomDegrees, double alpha);
bool performFisherTest(vector<double>& sample1, vector<double>& sample2, double& alpha, vector<double>& stdDvtns, vector<double>& freedomDegrees, vector<int>& samplesSize, vector<double>& means);
double calculateStudentStatEqualDisp(vector<double>& stdDvtns, vector<double>& freedomDegrees, vector<int>& samplesSize, vector<double>& means);
double calculateStudentStatNotEqualDisp(vector<double>& stdDvtns, vector<double>& freedomDegrees, vector<int>& samplesSize, vector<double>& means);
bool performStudentTest(bool fisherTestResult, double alpha, vector<double>& stdDvtns, vector<int>& samplesSize, vector<double>& means);

void writeOutput(const string& fileName,
    double alpha,
    const vector<double>& sample1,
    const vector<double>& sample2,
    const vector<double>& stdDvtns,
    const vector<double>& freedomDegrees,
    const vector<int>& samplesSize,
    const vector<double>& means,
    bool fisherResult,
    bool studentResult);

void readStartConfiguration(vector<double>& sample1, vector<double>& sample2, double& alpha);

#endif
