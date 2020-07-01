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
			data[x] = (int)(R * 0.3 + G * 0.59 + B * 0.11);//���ù�ʽ����Ҷ�ֵ����Ȩƽ������
			data[x] /= 255.0;
		}
	}
}
cv::Mat Integral(cv::Mat source)
{
	cv::Mat img(source.rows,source.cols , CV_32F);
	cv::Mat int_img(source.rows, source.cols, CV_32F);
	//�ҶȻ�
	cvtCOLOR1(source, img);
	int height = img.rows;
	int width = img.cols;
	int step = img.step / sizeof(float);
	float *data = (float *)img.data;
	float *i_data = (float *)int_img.data;  //��¼����
	float rs = 0.0f;
	//��һ��
	for (int j = 0; j<width; j++)
	{
		rs += data[j];//�ۻ���
		i_data[j] = rs;
	}
   //������
	for (int i = 1; i<height; ++i)
	{
		rs = 0.0f;
		for (int j = 0; j<width; ++j)
		{
			rs += data[i*step + j];//�ۻ���
			i_data[i*step + j] = rs + i_data[(i - 1)*step + j];//�ۻ���+��һ��
		}
	}
	return int_img;
}