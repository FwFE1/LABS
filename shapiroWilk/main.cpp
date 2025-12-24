#include "Source.h"

#include <iostream>

using namespace std;

const string inputFilePath = "Inp/shapirowilktest.inp";
const string outputFilePath = "Out/shapirowilktest.out";

const double alpha = 0.05;
const double generateMean = 1.0;
const double generateStdDev = 0.03;

int main(const int argc, const char** argv) {
    setlocale(LC_ALL, "Russian");

    try {
        Sample sample = generateInputFileWithNewSample(inputFilePath, generateMean, generateStdDev);
        cout << "������� ���� � �������� ������� �: " << inputFilePath << endl;
        std::vector<double> a_i = computeAi_viaIntegrals(sample.getSampleSize());
        cout << "������������ a_i ��������� ��� n = " << sample.getSampleSize() << endl;
        writeShapiroWilkResultToFile(outputFilePath, sample, a_i, alpha);
        cout << "���������� �������� �������� �: " << outputFilePath << endl;
    }
    catch (const std::exception& ex) {
        cerr << "������: " << ex.what() << endl;
    }

    return 0;
}
