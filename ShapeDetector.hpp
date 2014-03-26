#ifndef GRADIENTHANDLER_HPP_
#define GRADIENTHANDLER_HPP_

#include <opencv2/core/core.hpp>

class ShapeDetector {
public:
	ShapeDetector();
	enum TECHNS{ GTYPE_OCV, GTYPE_CUST, GTYPE_CUST2 };
	enum SHAPS{ SHAPE_CIR, SHAPE_TRI, SHAPE_SQR, SHAPE_OCT };

	void setImage( std::string path );

	void computeVoteImage( int shape, int method = GTYPE_CUST );
	void computeEquiMagnitude();
	void computeShapeResponse();

	cv::Mat getBaseImage();
	cv::Mat getGradientImage();
	cv::Mat getVoteImage();
	cv::Mat getMagEqImg();
	cv::Mat getShapeResponse();

	std::string getMethodName();

	~ShapeDetector();

protected:
	cv::Mat _baseImage;
	cv::Mat _gradientImage;
	cv::Mat _voteImage;
	cv::Mat _magEqImg;
	cv::Mat _shapeResponse;

	cv::Point** _equiImageData;

	std::string _imageName;
	int _shape;
	int _nSides;
	int _method;

private:
	void openCVGradient( float tanpi, int scale=1, int delta=0, int ddepth=CV_32F );
	void myCustomGradient(float tanpi);
	void myCustomGradient2(float tanpi);

	static const float TANPI8 = 0.41421356237;
	static const float TANPI4 = 1;
	static const float TANPI3 = 1.73205080757;
	static const float TANPI1 = 0;
	static const float RAD_TO_DEGREE = 57.295779513;

};

#endif
