#ifndef AUXILIAR_HPP_
#define AUXILIAR_HPP_

#include <opencv2/core/core.hpp>

#include <string>
#include <iostream>
#include <sstream>

class Auxiliar {
public:
	static void printImage(cv::Mat image)
	{
		std::cout << "Type: " << image.type() << std::endl;
		for( int i = 0; i < image.rows; i++ )
		{
			for( int j = 0; j < image.cols; j++ )
			{
				uchar v = image.at<uchar>(i,j);
				std::cout << v << "\t";
			}
			std::cout << std::endl;
		}
	}

private:
	Auxiliar(){}
};

#endif
