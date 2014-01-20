#include "GradientHandler.hpp"

#include <opencv2/imgproc/imgproc.hpp>

GradientHandler::GradientHandler()
{
	_baseImage = cv::Mat();
	_gradientImage = cv::Mat();
	_voteImage = cv::Mat();
}

GradientHandler::GradientHandler( cv::Mat baseImage )
{
	assert( baseImage.channels() == 1 && baseImage.type() == CV_8UC1 );

	_baseImage = baseImage;
	_gradientImage = cv::Mat( baseImage.rows-2, baseImage.cols-2, CV_8U );
	_voteImage = cv::Mat( baseImage.rows-2, baseImage.cols-2, CV_8U );
}

void GradientHandler::getGradient( int method )
{
	assert( !_baseImage.empty() );

	switch( method ) {
		case GTYPE_OCV:
			openCVGradient();
			break;
		case GTYPE_CUST:
			myCustomGradient();
			break;
		case GTYPE_CUST2:
			myCustomGradient2();
			break;
		default:
			break;
	}
}

cv::Mat GradientHandler::getGradientImage()
{
	return _gradientImage;
}

void GradientHandler::openCVGradient( int scale, int delta, int ddepth )
{
	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;

	// Gradient X
	cv::Sobel( _baseImage, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	// Gradient Y
	cv::Sobel( _baseImage, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	// Total Gradient (approximate)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, _gradientImage );

	// paper mentions threshold of 5% the max value found. Here we're using a fixed value.
	threshold( _gradientImage, _gradientImage, 50, 255, cv::THRESH_TOZERO );
}

void GradientHandler::myCustomGradient()
{
	cv::Mat auxGradientImage( _baseImage.rows-2, _baseImage.cols-2, CV_16U );

	std::pair<float,float> gradientVectorImage[_baseImage.rows-2][_baseImage.cols-2];

	short kernelX[3][3] = {	{1, 	0, 		-1},
			{2, 	0, 		-2},
			{1, 	0, 		-1}
	};

	short kernelY[3][3] = {	{1,		2, 		1},
			{0,		0, 		0},
			{-1,	-2, 	-1}
	};

	float maxVal = 0.0f;

	for( ushort i = 1; i < _baseImage.rows -1; i++ )
	{
		for( ushort j = 1; j < _baseImage.cols -1; j++ )
		{
			short vX = 0;
			short vY = 0;

			for( ushort ki = 0; ki < 3; ki++ )
			{
				short x = i+(ki-1);
				if( x >= 0 && x < _baseImage.rows )
				{
					for( ushort kj = 0; kj < 3; kj++ )
					{
						short y = j+(kj-1);
						if( y >= 0 && y < _baseImage.cols )
						{
							ushort valImg = _baseImage.at<uchar>(x,y);
							vX += valImg * kernelX[ki][kj];
							vY += valImg * kernelY[ki][kj];
						}
					}
				}
			}
			ushort magG = sqrt(pow(vX,2)+pow(vY,2));

			maxVal = (magG>maxVal)? magG : maxVal;

			auxGradientImage.at<ushort>(i-1,j-1) = magG;

			float angle = atan2(vY,vX);
			gradientVectorImage[i-1][j-1] = std::make_pair<float,float>( cos(angle), sin(angle) );
		}
	}

	float thresh = 0.10 * maxVal;

	for( int i = 0; i < auxGradientImage.rows; i++ )
	{
		for( int j = 0; j < auxGradientImage.cols; j++ )
		{
			if( auxGradientImage.at<ushort>(i,j) < thresh )
			{
				_gradientImage.at<uchar>(i,j) = 0;
			}
			else
			{
				_gradientImage.at<uchar>(i,j) = 255;
			}
		}
	}
}

void GradientHandler::myCustomGradient2()
{
	cv::Mat auxGradientImage( _baseImage.rows-2, _baseImage.cols-2, CV_16U );

	float maxVal = 0.0f;

	for( ushort i = 1; i < _baseImage.rows -1; i++ )
	{
		for( ushort j = 1; j < _baseImage.cols -1; j++ )
		{
			//short vY = (image.at<uchar>(i-1,j-1)+2*image.at<uchar>(i-1,j)+image.at<uchar>(i-1,j+1)) -
			//(image.at<uchar>(i+1,j-1)+2*image.at<uchar>(i+1,j)+image.at<uchar>(i+1,j+1));
			//short vX = (image.at<uchar>(i-1,j+1)+2*image.at<uchar>(i,j+1)+image.at<uchar>(i+1,j+1)) -
			//(image.at<uchar>(i-1,j-1)+2*image.at<uchar>(i,j-1)+image.at<uchar>(i+1,j-1));
			ushort magG =  abs( (_baseImage.at<uchar>(i-1,j-1)+2*_baseImage.at<uchar>(i-1,j)+_baseImage.at<uchar>(i-1,j+1)) -
					(_baseImage.at<uchar>(i+1,j-1)+2*_baseImage.at<uchar>(i+1,j)+_baseImage.at<uchar>(i+1,j+1)) ) +
							abs( (_baseImage.at<uchar>(i-1,j+1)+2*_baseImage.at<uchar>(i,j+1)+_baseImage.at<uchar>(i+1,j+1)) -
									(_baseImage.at<uchar>(i-1,j-1)+2*_baseImage.at<uchar>(i,j-1)+_baseImage.at<uchar>(i+1,j-1)) );

			maxVal = (magG>maxVal)? magG : maxVal;

			auxGradientImage.at<ushort>(i-1,j-1) = magG;
		}
	}

	float thresh = 0.10 * maxVal;

	for( int i = 0; i < auxGradientImage.rows; i++ )
	{
		for( int j = 0; j < auxGradientImage.cols; j++ )
		{
			if( auxGradientImage.at<ushort>(i,j) < thresh )
			{
				_gradientImage.at<uchar>(i,j) = 0;
			}
			else
			{
				_gradientImage.at<uchar>(i,j) = 255;
			}
		}
	}
}

GradientHandler::~GradientHandler()
{
	_baseImage.release();
	_gradientImage.release();
	_voteImage.release();
}
