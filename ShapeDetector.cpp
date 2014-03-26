#include "ShapeDetector.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

ShapeDetector::ShapeDetector()
{
	_shape = -1;
	_method = -1;
	_nSides = -1;
	_equiImageData = NULL;

	_minRadius = -1;
	_maxRadius = -1;
}

void ShapeDetector::setImage( std::string path )
{
	// load image as grayscale
	_baseImage = cv::imread( path, 0 );
	_imageName = path;
	_gradientImage.release();
	_voteImage.release();
}

cv::Mat ShapeDetector::getBaseImage()
{
	assert( !_baseImage.empty() );
	return _baseImage;
}

cv::Mat ShapeDetector::getGradientImage()
{
	assert( !_gradientImage.empty() );
	return _gradientImage;
}

cv::Mat ShapeDetector::getVoteImage()
{
	assert( !_voteImage.empty() );
	return _voteImage;
}

cv::Mat ShapeDetector::getMagEqImg()
{
	assert( !_magEqImg.empty() );
	return _magEqImg;
}

cv::Mat ShapeDetector::getShapeResponse()
{
	assert( !_shapeResponse.empty() );
	return _shapeResponse;
}

std::string ShapeDetector::getMethodName()
{
	std::string result;

	switch( _method ){
		case GTYPE_OCV:
			result = "OpenCV gradient";
			break;
		case GTYPE_CUST:
			result = "Custom method 1";
			break;
		case GTYPE_CUST2:
			result = "Custom method 2";
			break;
		default:
			break;
	}

	return result;
}

void ShapeDetector::computeVoteImage( int shape, int radius )
{
	assert( !_baseImage.empty() );

	float tanpi = 0.0f;
	_shape = shape;

	switch( shape ){
		case SHAPE_CIR:
			tanpi = TANPI1;
			_nSides = 1;
			break;
		case SHAPE_TRI:
			tanpi = TANPI3;
			_nSides = 3;
			break;
		case SHAPE_SQR:
			tanpi = TANPI4;
			_nSides = 4;
			break;
		case SHAPE_OCT:
			tanpi = TANPI8;
			_nSides = 8;
			break;
		default:
			break;
	}

	_voteImage = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_32S );

	_equiImageData = new cv::Point*[_baseImage.rows];

	for( int r = 0; r < _baseImage.rows; r++ )
	{
		_equiImageData[r] = new cv::Point[_baseImage.cols];
	}

	short w = round( radius * tanpi );
	short thresh = 50;

	for( int r = 0; r < _gradientImage.rows; r++ )
	{
		for( int c = 0; c < _gradientImage.cols; c++ )
		{
			if(_gradientImage.at<uchar>(r,c) >= thresh)
			{
				int mag = round(sqrt( pow(_gradY.at<float>(r,c),2)+ pow(_gradX.at<float>(r,c),2) ));
				float angle = atan2(_gradY.at<float>(r,c),_gradX.at<float>(r,c));
				short dx = round(radius*cos(angle));
				short dy = round(radius*sin(angle));
				float nangle = _nSides * angle;

				short vr = (r) + dy;
				short vc = (c) + dx;

				short xmin = vc - 2*w; if(xmin<0) xmin = 0;
				short xmax = vc + 2*w; if(xmax>=_voteImage.cols) xmax = _voteImage.cols-1;

				for( int i = xmin; i <= xmax; i++ ){
					if( i < _voteImage.cols && vr >= 0 && vr < _voteImage.rows )
					{
						if( i>=vc-w && i<=vc+w ){
							_voteImage.at<int>( vr, i ) += mag;

							_equiImageData[vr][i].x += round(cos(nangle));
							_equiImageData[vr][i].y += round(sin(nangle));
						}
						else{
							_voteImage.at<int>( vr, i ) -= mag;

							_equiImageData[vr][i].x += round(cos(-nangle));
							_equiImageData[vr][i].y += round(sin(-nangle));
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

void ShapeDetector::computeEquiMagnitude()
{
	assert( !_voteImage.empty() );
	_magEqImg = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_8U );

	for( int r = 0; r < _gradientImage.rows; r++ )
		for( int c = 0; c < _gradientImage.cols; c++ )
		{
			int res = sqrt( pow(_equiImageData[r][c].x,2)+pow(_equiImageData[r][c].y,2) );
			_magEqImg.at<uchar>(r,c) = res>255?255:res;
		}
}

void ShapeDetector::computeShapeResponse(int shape, int minRadius, int maxRadius)
{
	_shapeResponse = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_8U );
	openCVGradient();

	for( int i = minRadius; i <= maxRadius; i++ )
	{
		computeVoteImage(shape, i);
		computeEquiMagnitude();

		short w = round(i * TANPI4);
		float dem = pow(2*w*i,2);

		for( int r = 0; r < _baseImage.rows; r++ )
			for( int c = 0; c < _baseImage.cols; c++ )
			{
				int v = _voteImage.at<int>(r,c);
				int m = _magEqImg.at<uchar>(r,c);
				int num = v*m;
				if( num > 0 )
				{
					int res = round(num/dem);
					if(_shapeResponse.at<uchar>(r,c)+res>255)
						_shapeResponse.at<uchar>(r,c) = 255;
					else _shapeResponse.at<uchar>(r,c) += res;

					//_shapeResponse.at<int>(r,c) += res;
				}
			}
	}
}

void ShapeDetector::openCVGradient( int scale, int delta, int ddepth )
{
	cv::Mat abs_grad_x, abs_grad_y;
	_gradientImage = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_32S );

	// Gradient X
	cv::Sobel( _baseImage, _gradX, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( _gradX, abs_grad_x );

	// Gradient Y
	cv::Sobel( _baseImage, _gradY, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( _gradY, abs_grad_y );

	// Total Gradient (approximate)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, _gradientImage );
}

void ShapeDetector::myCustomGradient(float tanpi, int radius)
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
	short w = round(radius * tanpi);
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
			short dx = round(radius*cos(angle));
			short dy = round(radius*sin(angle));

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

void ShapeDetector::myCustomGradient2(float tanpi, int radius)
{
	cv::Mat auxGradientImage = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_16U );
	std::pair<short,short> gradientVectorImage[_baseImage.rows][_baseImage.cols];

	_gradientImage = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_8U );
	_voteImage = cv::Mat::zeros( _baseImage.rows, _baseImage.cols, CV_8U );

	float maxVal = 0.0f;
	short w = round(radius * tanpi);
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
			short dx = round(radius*cos(angle));
			short dy = round(radius*sin(angle));

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

void ShapeDetector::releaseEquiImageData()
{
	for( int i = 0; i < _baseImage.rows; ++i )
		delete [] _equiImageData[i];
	delete [] _equiImageData;
}

ShapeDetector::~ShapeDetector()
{
	_baseImage.release();
	_gradientImage.release();
	_voteImage.release();
	_magEqImg.release();
	_shapeResponse.release();

	releaseEquiImageData();
}
