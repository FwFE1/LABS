#include "Source.h"
#include <Windows.h>

int main(const int argc, const char** argv) {
	setlocale(LC_ALL, "Russian");
	SetConsoleOutputCP(65001); // 65001 = UTF-8
	srand(static_cast<unsigned int>(time(nullptr)));



	vector<double> sample1;
	vector<double> sample2;
	double alpha = 0.05;

	readStartConfiguration(sample1, sample2, alpha);

	vector<double> stdDvtns(2);
	vector<double> freedomDegrees;
	vector<int> samplesSize(2);
	vector<double> means(2);

	bool fisherResult = performFisherTest(sample1, sample2, alpha, stdDvtns, freedomDegrees, samplesSize, means);

	bool studentResult = performStudentTest(fisherResult, alpha, stdDvtns, samplesSize, means);

	writeOutput("Out/studentfisher.out",
		alpha,
		sample1,
		sample2,
		stdDvtns,
		freedomDegrees,
		samplesSize,
		means,
		fisherResult,
		studentResult);

	cout << "Результаты записаны в файл Out/studentfisher.out" << endl;

	return 0;
}