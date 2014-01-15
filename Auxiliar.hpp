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

		bool print = false;

		cv::Mat gradientImage( image.rows, image.cols, CV_32FC1 );

		short kernel[3][3] = {	{1, 0, -1},
								{2, 0, -2},
								{1, 0, -1}
							};

		for( int i = 0; i < image.rows; ++i )
		{
			for( int j = 0; j < image.cols; ++j )
			{
				float v = 0.0f;

				std::cout << "image x,y:" << i << "," << j << std::endl;
				for( int ki = 0; ki < 3; ki++ )
				{
					int x = i+(ki-1);
					if( x >= 0 && x<image.rows )
					{
						for( int kj = 0; kj < 3; kj++ )
						{
							int y = j+(kj-1);
							if( y>=0 && y<image.cols )
							{
								std::cout << "image neighbor x,y:" << x << "," << y;
								std::cout << " kernel value:" << kernel[ki][kj] << std::endl;
								v += ((short)image.at<uchar>(x,y)) * kernel[ki][kj];
							}
						}
					}
				}
				std::cout << std::endl;
				std::cout << std::endl;

				gradientImage.at<float>(i,j) = (float) v;
				if( print == true )
				{
					std::cout << v ;
					std::cout << "\t";
				}
		    }
			if( print == true) std::cout << std::endl;
		}

		return gradientImage;
	}

private:
	Auxiliar(){}
};

#endif
