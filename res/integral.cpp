#include "integral.h"
void cvtCOLOR1(cv::Mat src, cv::Mat dst)
{
	float R, G, B;
	for (int y = 0; y < src.rows; y++)
	{
		float* data = (float*)dst.ptr<uchar>(y);
		for (int x = 0; x < src.cols; x++)
		{
			B = src.at<cv::Vec3b>(y, x)[0];
			G = src.at<cv::Vec3b>(y, x)[1];
			R = src.at<cv::Vec3b>(y, x)[2];
			data[x] = (int)(R * 0.3 + G * 0.59 + B * 0.11);//利用公式计算灰度值（加权平均法）
			data[x] /= 255.0;
		}
	}
}
cv::Mat Integral(cv::Mat source)
{
	cv::Mat img(source.rows,source.cols , CV_32F);
	cv::Mat int_img(source.rows, source.cols, CV_32F);
	//灰度化
	cvtCOLOR1(source, img);
	int height = img.rows;
	int width = img.cols;
	int step = img.step / sizeof(float);
	float *data = (float *)img.data;
	float *i_data = (float *)int_img.data;  //记录积分
	float rs = 0.0f;
	//第一行
	for (int j = 0; j<width; j++)
	{
		rs += data[j];//累积和
		i_data[j] = rs;
	}
   //后续行
	for (int i = 1; i<height; ++i)
	{
		rs = 0.0f;
		for (int j = 0; j<width; ++j)
		{
			rs += data[i*step + j];//累积和
			i_data[i*step + j] = rs + i_data[(i - 1)*step + j];//累积和+上一行
		}
	}
	return int_img;
}