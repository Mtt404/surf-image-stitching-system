#pragma once
#include <cv.h>
#include "Mypoint.h"
#include <vector>
class ResponseLayer;
static const int OCTAVES = 5;//组数
static const int INTERVALS = 4;//层数
static const float THRES = 0.0004f;//阈值
static const int INIT_SAMPLE = 1;  //原始采样步长
class FastHessian {
public:
	//构造
	FastHessian(cv::Mat *img,
		vector<Mypoint> &mpts,
		const int octaves = OCTAVES,
		const int intervals = INTERVALS,
		const int init_sample = INIT_SAMPLE,
		const float thres = THRES);

	//! 析构
	~FastHessian();
	//! 保存参数
	void saveParameters(const int octaves,
		const int intervals,
		const int init_sample,
		const float thres);
	//设置或重置积分图像
	void setIntImage(cv::Mat *img);
	//找到特征点
	void getMypoints(int flag);

private:
	//!  构建响应图
	void buildResponseMap();
	//!  计算所提供层的响应
	void buildResponseLayer(ResponseLayer *r);
	//! 3x3x3 非极大值抑制
	int isExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
	int isExtremum1(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
	int isExtremum2(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
	//! 插值函数，逼近真实极值
	void interpolateExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
	//真正极值点的偏差
	void interpolateStep(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b,double* xi, double* xr, double* xc);
	//3D偏导
	cv::Mat deriv3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
	//3D二阶导
	cv::Mat hessian3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
	//积分图像及其参数
	cv::Mat *img;
	int i_width, i_height;
	//特征向量
	vector<Mypoint> &mpts;
	//尺度空间
	vector<ResponseLayer *> responseMap;
	//尺度空间图像集组数
	int octaves;
	//尺度空间图像集每组层数
	int intervals;
	//原始采样步长
	int init_sample;
	//极值检测阈值
	float thresh;
};

