#include "GradientHandler.hpp"

#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

GradientHandler::GradientHandler()
{
	_baseImage = cv::Mat();
}

GradientHandler::GradientHandler( cv::Mat baseImage )
{
	assert( baseImage.channels() == 1 && baseImage.type() == CV_8UC1 );

	_baseImage = baseImage;
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
	assert( !_gradientImage.empty() );
	return _gradientImage;
}

cv::Mat GradientHandler::getVoteImage()
{
	assert( !_voteImage.empty() );
	return _voteImage;
}

void GradientHandler::openCVGradient( int scale, int delta, int ddepth )
{
	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;

	_gradientImage = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_8U );
	_voteImage = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_8U );

	short rd = 5;
	short w = round( rd * TANPI4 );
	short thresh = 50;

	// Gradient X
	cv::Sobel( _baseImage, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	// Gradient Y
	cv::Sobel( _baseImage, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	// Total Gradient (approximate)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, _gradientImage );

	for( int r = 0; r < _gradientImage.rows; r++ )
	{
		for( int c = 0; c < _gradientImage.cols; c++ )
		{
			if(_gradientImage.at<uchar>(r,c) >= thresh)
			{
				float angle = atan2(grad_y.at<float>(r,c),grad_x.at<float>(r,c));
				short dx = round(rd*cos(angle));
				short dy = round(rd*sin(angle));

				short vr = (r) + dy;
				short vc = (c) + dx;

				short xmin = vc - 2*w; if(xmin<0) xmin = 0;
				short xmax = vc + 2*w; if(xmax>_voteImage.rows-1) xmax = _voteImage.rows-1;

				for( int i = xmin; i <= xmax; i++ ){
					if( i < _voteImage.cols && vr >= 0 && vr < _voteImage.rows )
					{
						if( i>=vc-w && i<=vc+w ){
							if( _voteImage.at<uchar>( vr, i) < 245 ) _voteImage.at<uchar>( vr, i ) += 10;
						}
						else{
							if( _voteImage.at<uchar>( vr, i) > 9 ) _voteImage.at<uchar>( vr, i ) -= 10;
						}
					}
				}
			}
			else
			{
				_gradientImage.at<uchar>(r,c) = 0;
			}
		}
	}
}

void GradientHandler::myCustomGradient()
{
	cv::Mat auxGradientImage = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_16U );
	std::pair<short,short> gradientVectorImage[_baseImage.rows][_baseImage.cols];

	_gradientImage = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_8U );
	_voteImage = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_8U );

	short kernelX[3][3] = {	{-1, 	0, 		1},
							{-2, 	0, 		2},
							{-1, 	0, 		1}
	};

	short kernelY[3][3] = {	{-1,	-2,		-1},
							{0,		0,		0},
							{1,		2,		1}
	};

	float maxVal = 0.0f;
	float rd = 5;
	short w = round(rd * TANPI4);
	float threshRatio = 0.15f;

	for( ushort r = 1; r < _baseImage.rows - 1; r++ )
	{
		for( ushort c = 1; c < _baseImage.cols - 1; c++ )
		{
			short vX = 0;
			short vY = 0;

			for( ushort ki = 0; ki < 3; ki++ )
			{
				short y = r+(ki-1);
				if( y >= 0 && y < _baseImage.rows )
				{
					for( ushort kj = 0; kj < 3; kj++ )
					{
						short x = c+(kj-1);
						if( x >= 0 && x < _baseImage.cols )
						{
							ushort valImg = _baseImage.at<uchar>(y,x);
							vX += valImg * kernelX[ki][kj];
							vY += valImg * kernelY[ki][kj];
						}
					}
				}
			}
			ushort magG = sqrt(pow(vX,2)+pow(vY,2));

			maxVal = (magG>maxVal)? magG : maxVal;

			auxGradientImage.at<ushort>(r,c) = magG;

			float angle = atan2(vY,vX);
			short dx = round(rd*cos(angle));
			short dy = round(rd*sin(angle));

			gradientVectorImage[r][c] = std::make_pair<short,short>( dx, dy );
		}
	}

	float thresh = threshRatio * maxVal;

	for( int r = 0; r < auxGradientImage.rows; r++ )
	{
		for( int c = 0; c < auxGradientImage.cols; c++ )
		{
			if( auxGradientImage.at<ushort>(r,c) < thresh )
			{
				_gradientImage.at<uchar>(r,c) = 0;
			}
			else
			{
				_gradientImage.at<uchar>(r,c) = 255;

				float dx = gradientVectorImage[r][c].first;
				float dy = gradientVectorImage[r][c].second;

				short vr = (r) + dy;
				short vc = (c) + dx;

				short xmin = vc - 2*w; if(xmin<0) xmin = 0;
				short xmax = vc + 2*w; if(xmax>_voteImage.rows-1) xmax = _voteImage.rows-1;

				for( int i = xmin; i <= xmax; i++ ){
					if( i < _voteImage.cols && vr >= 0 && vr < _voteImage.rows )
					{
						if( i>=vc-w && i<=vc+w ){
							if( _voteImage.at<uchar>( vr, i) < 245 ) _voteImage.at<uchar>( vr, i ) += 10;
						}
						else{
							if( _voteImage.at<uchar>( vr, i) > 9 ) _voteImage.at<uchar>( vr, i ) -= 10;
						}
					}
				}
			}
		}
	}
}

void GradientHandler::myCustomGradient2()
{
	cv::Mat auxGradientImage = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_16U );
	std::pair<short,short> gradientVectorImage[_baseImage.rows][_baseImage.cols];

	_gradientImage = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_8U );
	_voteImage = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_8U );

	float maxVal = 0.0f;
	float rd = 5;
	short w = round(rd * TANPI4);
	float threshRatio = 0.15f;

	for( ushort r = 1; r < _baseImage.rows - 1; r++ )
	{
		for( ushort c = 1; c < _baseImage.cols - 1; c++ )
		{
			short vY = (_baseImage.at<uchar>(r+1,c-1)+2*_baseImage.at<uchar>(r+1,c)+_baseImage.at<uchar>(r+1,c+1)) -
					(_baseImage.at<uchar>(r-1,c-1)+2*_baseImage.at<uchar>(r-1,c)+_baseImage.at<uchar>(r-1,c+1));
			short vX = (_baseImage.at<uchar>(r-1,c+1)+2*_baseImage.at<uchar>(r,c+1)+_baseImage.at<uchar>(r+1,c+1)) -
					(_baseImage.at<uchar>(r-1,c-1)+2*_baseImage.at<uchar>(r,c-1)+_baseImage.at<uchar>(r+1,c-1));
			ushort magG =  abs(vY) + abs(vX);

			maxVal = (magG>maxVal) ? magG : maxVal;

			auxGradientImage.at<ushort>(r,c) = magG;

			float angle = atan2(vY,vX) * RAD_TO_DEGREE;
			short dx = round(rd*cos(angle));
			short dy = round(rd*sin(angle));

			std::cout << r << "," << c << ": " << vY << ";" << vX << "->" << angle << "->" << dx << ";"<<dy<< "\t";

			gradientVectorImage[r][c] = std::make_pair<short,short>( dx, dy );
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	float thresh = threshRatio * maxVal;

	for( int r = 0; r < auxGradientImage.rows; r++ )
	{
		for( int c = 0; c < auxGradientImage.cols; c++ )
		{
			if( auxGradientImage.at<ushort>(r,c) < thresh )
			{
				_gradientImage.at<uchar>(r,c) = 0;
			}
			else
			{
				_gradientImage.at<uchar>(r,c) = 255;
				float dx = gradientVectorImage[r][c].first;
				float dy = gradientVectorImage[r][c].second;

				std::cout << r << "," << c << "->" << dx << ";"<<dy<< "\t";

				short vr = (r) + dy;
				short vc = (c) + dx;

				short xmin = vc - 2*w;
				if(xmin<0) xmin = 0;
				short xmax = vc + 2*w;
				if(xmax>_voteImage.rows-1) xmax = _voteImage.rows-1;

				for( int i = xmin; i <= xmax; i++ ){
					if( i < _voteImage.cols && vr >= 0 && vr < _voteImage.rows )
					{
						if( i>=vc-w && i<=vc+w ){
							if( _voteImage.at<uchar>( vr, i) < 245 ) _voteImage.at<uchar>( vr, i ) += 10;
						}
						else{
							if( _voteImage.at<uchar>( vr, i) > 9 ) _voteImage.at<uchar>( vr, i ) -= 10;
						}
					}
				}
			}
		}
		std::cout << std::endl;
	}
}

GradientHandler::~GradientHandler()
{
	_baseImage.release();
	_gradientImage.release();
	_voteImage.release();
}
