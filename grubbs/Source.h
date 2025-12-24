#ifndef SOURCE_H
#define SOURCE_H

#include <vector>
#include <algorithm>
#include <random>
#include <climits>
#include <ctime>
#include <iostream>
#include <fstream>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>



using namespace std;
using namespace boost::math;

void showVector(vector <double>& sample);
double maxVector(vector <double>& vect);
double minVector(vector <double>& vect);

double normPPF(double p);
double studentPPF(double alpha, const int freedomDegree);

void insertSus(vector<double>& sample, double alpha = 0.05);
void generateSample(vector <double>& sample, double& alpha, int& state, double& a, double& s);

void readStartConfiguration(const string& fileName,
    vector<double>& sample,
    double& alpha,
    int& state,
    double& a,
    double& s);

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
    bool h0Accepted);

double calculateMean(vector <double>& sample);
double calculateStandartDeviation(vector <double>& sample, double mean);
double calculateGrubbsStat(vector <double>& sample, double mean, double standartDeviation, int state = 0);
double calculateUAlpha(const int sampleSize, double alpha, bool state = false);
bool performGrubbsTest(vector <double>& sample, double alpha, int state, double a, double s, string FileName);

#endif // SOURCE_H