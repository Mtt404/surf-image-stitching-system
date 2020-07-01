#pragma once
#include <algorithm> 
#include <cv.h>
cv::Mat Integral(cv::Mat img);//ָ�뺯�� ��ȡ����ͼ
//���ͻ��ֽ�� --����������Ƶ�����ú�����ջ�ڴ��ظ����������������ģ�inline�����Ķ������ͷ�ļ���
inline float BoxIntegral(cv::Mat *img, int row, int col, int rows, int cols)
{
	float *data = (float *)img->data;
	int step = img->step / sizeof(float);
	// ��/�����һ����Ϊ��/�а���
	int r1 = std::min(row, img->rows) - 1;//��ֹԽ��
	int c1 = std::min(col, img->cols) - 1;
	int r2 = std::min(row + rows, img->rows) - 1;
	int c2 = std::min(col + cols, img->cols) - 1;
	float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
	if (r1 >= 0 && c1 >= 0) A = data[r1 * step + c1];//��֤��Խ�磬����ͼ����
	if (r1 >= 0 && c2 >= 0) B = data[r1 * step + c2];
	if (r2 >= 0 && c1 >= 0) C = data[r2 * step + c1];
	if (r2 >= 0 && c2 >= 0) D = data[r2 * step + c2];
	return std::max(0.f, A - B - C + D);
}

