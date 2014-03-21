#include "auxiliar/Auxiliar.hpp"
#include "GradientHandler.hpp"

#include <opencv2/highgui/highgui.hpp>

#include <cmath>

int main(int argc, char** argv) {
	cv::Mat image, grad, vote;
	GradientHandler gh;

	gh = GradientHandler();
	gh.setImage("tests/test6.jpg");

	//-----------------------------------Calculate_MyGradient_Image-----------------------------------//
	clock_t t;
	t = clock();

	gh.computeGradient( GradientHandler::GTYPE_OCV );

	t = clock() - t;
	float calcDuration = ( (float) t ) / CLOCKS_PER_SEC;

	std::cout << "Custom gradient (technique 1) image obtained in " << calcDuration << " seconds." << std::endl;

	//-----------------------------------Calculate_MyGradient_Image-----------------------------------//
	/**t = clock();

	grad2 = Auxiliar::myCustomGradient2( image );

	t = clock() - t;
	calcDuration = ( (float) t ) / CLOCKS_PER_SEC;

	std::cout << "Custom gradient (technique 2) image obtained in " << calcDuration << " seconds." << std::endl;*/

	//---------------------------------Calculate_OpenCVGradient_Image---------------------------------//
	/**t = clock();

	grad3 = Auxiliar::getGradient( image );

	t = clock() - t;
	calcDuration = ( (float) t ) / CLOCKS_PER_SEC;

	std::cout << "OpenCV gradient image obtained in " << calcDuration << " seconds." << std::endl;*/

	//-----------------------------------Calculate_Gradient_Images------------------------------------//

	image = gh.getBaseImage();
	grad = gh.getGradientImage();
	vote = gh.getVoteImage();

	cv::namedWindow("Original", CV_WINDOW_NORMAL);
	cv::imshow("Original", image);

	cv::namedWindow("Gradient", CV_WINDOW_NORMAL);
	cv::imshow("Gradient", grad);

	cv::namedWindow("Vote", CV_WINDOW_NORMAL);
	cv::imshow("Vote", vote);

	/**cv::namedWindow("MyGradient (technique 1)", CV_WINDOW_NORMAL);
	cv::imshow("MyGradient (technique 1)", grad1);

	cv::namedWindow("MyGradient (technique 2)", CV_WINDOW_NORMAL);
	cv::imshow("MyGradient (technique 2)", grad2);

	cv::namedWindow("OpenCVGradient", CV_WINDOW_NORMAL);
	cv::imshow("OpenCVGradient", grad3);*/

	cv::waitKey(0);
}
