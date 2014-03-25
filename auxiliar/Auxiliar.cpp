#include "Auxiliar.hpp"

#include <iomanip>

void Auxiliar::printImage(cv::Mat image)
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

void Auxiliar::printFImage(cv::Mat image)
{
	std::cout << "Type: " << image.type() << std::endl;
	for( int i = 0; i < image.rows; i++ )
	{
		for( int j = 0; j < image.cols; j++ )
		{
			float v = image.at<float>(i,j);
			std::cout << std::fixed << std::setw(9)
	        << std::setprecision(6) << v << "   ";
		}
		std::cout << std::endl;
	}
}
