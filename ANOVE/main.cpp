#include "Source.h"

#include <iostream>

using namespace std;

const int numberOfSamples = 3;
const double alpha = 0.05;
const string inputFile = "Inp/annova.inp";
const string outputFile = "Out/annova.out";

int main(const int argc, const char** argv[]) {
	setlocale(LC_ALL, "Russian");

	vector<Sample> samples;
	generateInputFileIfEmpty(inputFile, numberOfSamples, samples);

	double generalMean = calculateGeneralMean(samples);
	cout << "Общее среднее a = " << generalMean << endl;

	double outerDispersion = calculateOuterDispersion(samples, generalMean);
	cout << "Межгрупповая дисперсия s_1 = " << outerDispersion << endl;

	double innerDispersion = calculateInnerDispersion(samples);
	cout << "Внутригрупповая дисперсия s_2 = " << innerDispersion << endl;

	double F = calculateFisherStat(outerDispersion, innerDispersion);
	cout << "Наблюдаемое значение F = " << innerDispersion << endl;

	pair<int, int> freedomDegrees = calculateFreedomDegreees(samples);
	cout << "Степени свободы f1, f2 = " << freedomDegrees.first << ", " << freedomDegrees.second << endl;

	double F_alpha = calculateFAlpha(freedomDegrees, alpha);
	cout << "Критическое значение F_alpha = " << F_alpha << endl;

	if (F <= F_alpha) cout << "Нулевая гипотеза H_0 принимается" << endl;
	else cout << "Нулевая гипотеза H_0 отвергается" << endl;

	writeAnovaToFile(outputFile, alpha, samples);

	return 0;
}