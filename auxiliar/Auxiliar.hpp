#ifndef AUXILIAR_HPP_
#define AUXILIAR_HPP_

#include <opencv2/core/core.hpp>

#include <string>
#include <iostream>
#include <sstream>

class Auxiliar {
public:
	static void printImage(cv::Mat image);

private:
	Auxiliar(){}
};

#endif
