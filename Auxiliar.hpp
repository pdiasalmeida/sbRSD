#ifndef AUXILIAR_HPP_
#define AUXILIAR_HPP_

#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <iostream>
#include <sstream>

class Auxiliar {
public:
	static cv::Mat getGradient( cv::Mat image, int scale=1, int delta=0, int ddepth=CV_32F )
	{
		cv::Mat gradientImage;

		cv::Mat grad_x, grad_y;
		cv::Mat abs_grad_x, abs_grad_y;

		// Gradient X
		cv::Sobel( image, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
		convertScaleAbs( grad_x, abs_grad_x );

		// Gradient Y
		cv::Sobel( image, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
		convertScaleAbs( grad_y, abs_grad_y );

		// Total Gradient (approximate)
		addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradientImage );

		// paper mentions threshold of 5% the max value found. Here we're using a fixed value.
		threshold( gradientImage, gradientImage, 50, 255, cv::THRESH_TOZERO );


		return gradientImage;
	}

	static cv::Mat myCustomGradient(cv::Mat image)
	{
		assert( image.channels() == 1 && image.type() == CV_8UC1 );

		cv::Mat gradientImage( image.rows-2, image.cols-2, CV_16U );
		cv::Mat gradientImageOut( image.rows-2, image.cols-2, CV_8U );

		short kernelX[3][3] = {	{1, 	0, 		-1},
								{2, 	0, 		-2},
								{1, 	0, 		-1}
							};

		short kernelY[3][3] = {	{1,		2, 		1},
								{0,		0, 		0},
								{-1,	-2, 	-1}
							};

		float maxVal = 0.0f;

		for( ushort i = 1; i < image.rows -1; i++ )
		{
			for( ushort j = 1; j < image.cols -1; j++ )
			{
				short vX = 0;
				short vY = 0;

				for( ushort ki = 0; ki < 3; ki++ )
				{
					short x = i+(ki-1);
					if( x >= 0 && x<image.rows )
					{
						for( ushort kj = 0; kj < 3; kj++ )
						{
							short y = j+(kj-1);
							if( y>=0 && y<image.cols )
							{
								ushort valImg = image.at<uchar>(x,y);
								vX += valImg * kernelX[ki][kj];
								vY += valImg * kernelY[ki][kj];
							}
						}
					}
				}
				ushort magG = sqrt(pow(vX,2)+pow(vY,2));

				maxVal = (magG>maxVal)? magG : maxVal;

				gradientImage.at<ushort>(i-1,j-1) = magG;
		    }
		}

		float thresh = 0.05 * maxVal;

		for( int i = 0; i < gradientImage.rows; i++ )
		{
			for( int j = 0; j < gradientImage.cols; j++ )
			{
				if( gradientImage.at<ushort>(i,j) < thresh )
				{
					gradientImageOut.at<uchar>(i,j) = 0;
				}
				else
				{
					gradientImageOut.at<uchar>(i,j) = (gradientImage.at<ushort>(i,j) / maxVal) * 255;
				}
			}
		}

		return gradientImageOut;
	}

	static cv::Mat myCustomGradient2(cv::Mat image)
		{
			assert( image.channels() == 1 && image.type() == CV_8UC1 );

			cv::Mat gradientImage( image.rows-2, image.cols-2, CV_16U );
			cv::Mat gradientImageOut( image.rows-2, image.cols-2, CV_8U );

			float maxVal = 0.0f;

			for( ushort i = 1; i < image.rows -1; i++ )
			{
				for( ushort j = 1; j < image.cols -1; j++ )
				{
					ushort magG =  abs( (image.at<uchar>(i-1,j-1)+2*image.at<uchar>(i-1,j)+image.at<uchar>(i-1,j+1)) -
							(image.at<uchar>(i+1,j-1)+2*image.at<uchar>(i+1,j)+image.at<uchar>(i+1,j+1)) ) +
									abs( (image.at<uchar>(i-1,j+1)+2*image.at<uchar>(i,j+1)+image.at<uchar>(i+1,j+1)) -
											(image.at<uchar>(i-1,j-1)+2*image.at<uchar>(i,j-1)+image.at<uchar>(i+1,j-1)) );

					maxVal = (magG>maxVal)? magG : maxVal;

					gradientImage.at<ushort>(i-1,j-1) = magG;
			    }
			}

			float thresh = 0.05 * maxVal;

			for( int i = 0; i < gradientImage.rows; i++ )
			{
				for( int j = 0; j < gradientImage.cols; j++ )
				{
					if( gradientImage.at<ushort>(i,j) < thresh )
					{
						gradientImageOut.at<uchar>(i,j) = 0;
					}
					else
					{
						gradientImageOut.at<uchar>(i,j) = (gradientImage.at<ushort>(i,j) / maxVal) * 255;
					}
				}
			}

			return gradientImageOut;
		}

	static void printImage(cv::Mat image)
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

private:
	Auxiliar(){}
};

#endif
