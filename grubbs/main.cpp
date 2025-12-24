#include "Source.h";




int main(const int argc, const char** argv) {
	srand(time(nullptr));
	setlocale(LC_ALL, "rus");
	
	string inputFileName = "grubbs.inp";
	string OutputFileName = "grubbs.out";

	vector<double> sample;
	double alpha = 0;
	int state = 0;
	double a = 0.0;
	double s = 0.0;

	readStartConfiguration(inputFileName, sample, alpha, state, a, s);

	cout << endl << "Исходная выборка::" << endl;
	showVector(sample);
	cout << endl;

	cout << "Параметры нормального распределения (из файла / сгенерированные):"  << endl;
	cout << "a = " << a << endl;
	cout << "s = " << s << endl;
	cout << endl;

	bool grubbsResult = performGrubbsTest(sample, alpha, state, a, s, OutputFileName);
	
	return 0;
}