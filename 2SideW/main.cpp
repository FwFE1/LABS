#include "Source.h"
#include <iostream>

int main()
{
    setlocale(LC_ALL, "Russian");
    try {
        generateTwoSidedWilcoxonInp(0.05);
        runTwoSidedWilcoxonFromFile();
    }
    catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << std::endl;
    }
}
