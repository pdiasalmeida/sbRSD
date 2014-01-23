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
	cv::Mat getVoteImage();

	~GradientHandler();

protected:
	cv::Mat _baseImage;
	cv::Mat _gradientImage;
	cv::Mat _voteImage;
	std::pair<float,float> _equiImage;

private:
	void openCVGradient( int scale=1, int delta=0, int ddepth=CV_32F );
	void myCustomGradient();
	void myCustomGradient2();

	static const float TANPI8 = 0.41421356237;
	static const float TANPI4 = 1;
	static const float TANPI3 = 1.73205080757;
	static const float TANPI1 = 0;
	static const float RAD_TO_DEGREE = 57.295779513;

};

#endif
