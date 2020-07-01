#pragma once
#include <cv.h>
#include "Mypoint.h"
#include <vector>
#include <highgui.h>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include "integral.h"
#include "fasthessian.h"
#include "surf.h"
inline void surfDetDes(int flag,cv::Mat *img,  //输入图像 
	std::vector<Mypoint> &mpts, //特征点集 
	int octaves = OCTAVES, // 尺度空间图像集组数 
	int intervals = INTERVALS, // 尺度空间图像集每组层数 
	int init_sample = INIT_SAMPLE, // 初始采样步长 
	float thres = THRES  )// 极值响应阈值
{
	// 获取积分图像
	cv::Mat int_img = Integral(*img);
	//快速海塞检测器
	FastHessian fh(&int_img, mpts, octaves, intervals, init_sample, thres);
	// 获取特征点
	fh.getMypoints(flag);
	//构建SURF类
	Surf desc(&int_img, mpts);
	//获取特征描述
	desc.Surf::getDescriptors();
}
inline void surfDet(cv::Mat *img, //输入图像 
	std::vector<Mypoint> &mpts, //特征点集 
	int octaves = OCTAVES, // 尺度空间图像集组数
	int intervals = INTERVALS,// 尺度空间图像集每组层数 
	int init_sample = INIT_SAMPLE, // 初始采样步长 
	float thres = THRES )// 极值响应阈值
{
	// 获取积分图像
	cv::Mat int_img = Integral(*img);
	//快速海塞检测器
	FastHessian fh(&int_img, mpts, octaves, intervals, init_sample, thres);
	// 获取特征点
	fh.getMypoints(0);	
}
//绘出特征点
void drawPoint1(cv::Mat img, Mypoint &mpt);
void getMatches(MpVec &mpts1, MpVec &mpts2, MpPairVec &matches);

