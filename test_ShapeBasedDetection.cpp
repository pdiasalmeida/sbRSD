#include "Auxiliar.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cmath>

int main(int argc, char** argv) {
	cv::Mat image, grad;

	// load image as grayscale
	image = cv::imread( "tests/test2.jpg", 0 );

	//-----------------------------------Calculate_Gradient_Image-----------------------------------//
	clock_t t;
	t = clock();

	grad = Auxiliar::myCustomGradient( image );

	t = clock() - t;
	float calcDuration = ( (float) t ) / CLOCKS_PER_SEC;

	std::cout << "Gradient image obtained in " << calcDuration << " seconds." << std::endl;

	//-----------------------------------Calculate_Gradient_Image-----------------------------------//

	//cv::namedWindow("Gradient", CV_WINDOW_NORMAL);
	//cv::imshow("Gradient", grad);

	cv::waitKey(0);
}
