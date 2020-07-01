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
inline void surfDetDes(int flag,cv::Mat *img,  //����ͼ�� 
	std::vector<Mypoint> &mpts, //�����㼯 
	int octaves = OCTAVES, // �߶ȿռ�ͼ������ 
	int intervals = INTERVALS, // �߶ȿռ�ͼ��ÿ����� 
	int init_sample = INIT_SAMPLE, // ��ʼ�������� 
	float thres = THRES  )// ��ֵ��Ӧ��ֵ
{
	// ��ȡ����ͼ��
	cv::Mat int_img = Integral(*img);
	//���ٺ��������
	FastHessian fh(&int_img, mpts, octaves, intervals, init_sample, thres);
	// ��ȡ������
	fh.getMypoints(flag);
	//����SURF��
	Surf desc(&int_img, mpts);
	//��ȡ��������
	desc.Surf::getDescriptors();
}
inline void surfDet(cv::Mat *img, //����ͼ�� 
	std::vector<Mypoint> &mpts, //�����㼯 
	int octaves = OCTAVES, // �߶ȿռ�ͼ������
	int intervals = INTERVALS,// �߶ȿռ�ͼ��ÿ����� 
	int init_sample = INIT_SAMPLE, // ��ʼ�������� 
	float thres = THRES )// ��ֵ��Ӧ��ֵ
{
	// ��ȡ����ͼ��
	cv::Mat int_img = Integral(*img);
	//���ٺ��������
	FastHessian fh(&int_img, mpts, octaves, intervals, init_sample, thres);
	// ��ȡ������
	fh.getMypoints(0);	
}
//���������
void drawPoint1(cv::Mat img, Mypoint &mpt);
void getMatches(MpVec &mpts1, MpVec &mpts2, MpPairVec &matches);

