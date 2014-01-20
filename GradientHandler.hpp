#ifndef GRADIENTHANDLER_HPP_
#define GRADIENTHANDLER_HPP_

#include <opencv2/core/core.hpp>

class GradientHandler {
public:
	GradientHandler();
	GradientHandler( cv::Mat baseImage );

	enum TECHNS{ GTYPE_OCV, GTYPE_CUST, GTYPE_CUST2 };

	void getGradient( int method = GTYPE_CUST );

	cv::Mat getGradientImage();

	~GradientHandler();

protected:
	cv::Mat _baseImage;
	cv::Mat _gradientImage;
	cv::Mat _voteImage;

private:
	void openCVGradient( int scale=1, int delta=0, int ddepth=CV_32F );
	void myCustomGradient();
	void myCustomGradient2();

};

#endif
