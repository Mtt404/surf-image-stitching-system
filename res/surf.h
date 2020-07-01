#pragma once
#include <cv.h>
#include "Mypoint.h"
#include "integral.h"
#include <vector>

class Surf {
public:
	//���캯��
	Surf(cv::Mat *img, MpVec &mpts);
	//���������������
	void getDescriptors();
private:
	//�������������
	void getOrientation();
	//! Get the descriptor. See Agrawal ECCV 08 ��ȡ ��������*******
	void getDescriptor();
	//! ����x��y����2d��˹ֵ
	inline float gaussian(int x, int y, float sigma);
	inline float gaussian(float x, float y, float sigma);
	//! ����x��y�����HaarС����Ӧ
	inline float haarX(int row, int column, int size);
	inline float haarY(int row, int column, int size);
	//��[X Y]������������+ ve x���ȡ�Ƕ�
	float getAngle(float X, float Y);
	cv::Mat *img;
	MpVec &mpts;
	//��ǰ�������
	int index;
};