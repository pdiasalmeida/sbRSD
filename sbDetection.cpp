#include "auxiliar/Auxiliar.hpp"
#include "ShapeDetector.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>

int main(int argc, char** argv) {
	cv::Mat image, grad, vote;
	ShapeDetector sd;

	sd = ShapeDetector();

	clock_t t;
	t = clock();

	sd.setImage("tests/test6.jpg");
	sd.computeGradient( ShapeDetector::SHAPE_SQR, ShapeDetector::GTYPE_OCV );

	t = clock() - t;
	float calcDuration = ( (float) t ) / CLOCKS_PER_SEC;

	std::cout << "Shape detection, using the method '" << sd.getMethodName()
			<< "' computed in  " << calcDuration << " seconds." << std::endl;

	image = sd.getBaseImage();
	grad = sd.getGradientImage();
	vote = sd.getVoteImage();

	Auxiliar::printFImage(sd.getGradientAngles());

	cv::namedWindow("Original", CV_WINDOW_NORMAL);
	cv::imshow("Original", image);

	cv::namedWindow("Gradient", CV_WINDOW_NORMAL);
	cv::imshow("Gradient", grad);

	cv::namedWindow("Vote", CV_WINDOW_NORMAL);
	cv::imshow("Vote", vote);

	cv::waitKey(0);
}
