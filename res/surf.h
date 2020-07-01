#pragma once
#include <cv.h>
#include "Mypoint.h"
#include "integral.h"
#include <vector>

class Surf {
public:
	//构造函数
	Surf(cv::Mat *img, MpVec &mpts);
	//获得所有特征描述
	void getDescriptors();
private:
	//获得特征主方向
	void getOrientation();
	//! Get the descriptor. See Agrawal ECCV 08 获取 特征描述*******
	void getDescriptor();
	//! 计算x，y处的2d高斯值
	inline float gaussian(int x, int y, float sigma);
	inline float gaussian(float x, float y, float sigma);
	//! 计算x和y方向的Haar小波响应
	inline float haarX(int row, int column, int size);
	inline float haarY(int row, int column, int size);
	//从[X Y]给定的向量的+ ve x轴获取角度
	float getAngle(float X, float Y);
	cv::Mat *img;
	MpVec &mpts;
	//当前点的索引
	int index;
};