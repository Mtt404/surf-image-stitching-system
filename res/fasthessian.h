#pragma once
#include <cv.h>
#include "Mypoint.h"
#include <vector>
class ResponseLayer;
static const int OCTAVES = 5;//����
static const int INTERVALS = 4;//����
static const float THRES = 0.0004f;//��ֵ
static const int INIT_SAMPLE = 1;  //ԭʼ��������
class FastHessian {
public:
	//����
	FastHessian(cv::Mat *img,
		vector<Mypoint> &mpts,
		const int octaves = OCTAVES,
		const int intervals = INTERVALS,
		const int init_sample = INIT_SAMPLE,
		const float thres = THRES);

	//! ����
	~FastHessian();
	//! �������
	void saveParameters(const int octaves,
		const int intervals,
		const int init_sample,
		const float thres);
	//���û����û���ͼ��
	void setIntImage(cv::Mat *img);
	//�ҵ�������
	void getMypoints(int flag);

private:
	//!  ������Ӧͼ
	void buildResponseMap();
	//!  �������ṩ�����Ӧ
	void buildResponseLayer(ResponseLayer *r);
	//! 3x3x3 �Ǽ���ֵ����
	int isExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
	int isExtremum1(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
	int isExtremum2(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
	//! ��ֵ�������ƽ���ʵ��ֵ
	void interpolateExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
	//������ֵ���ƫ��
	void interpolateStep(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b,double* xi, double* xr, double* xc);
	//3Dƫ��
	cv::Mat deriv3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
	//3D���׵�
	cv::Mat hessian3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
	//����ͼ�������
	cv::Mat *img;
	int i_width, i_height;
	//��������
	vector<Mypoint> &mpts;
	//�߶ȿռ�
	vector<ResponseLayer *> responseMap;
	//�߶ȿռ�ͼ������
	int octaves;
	//�߶ȿռ�ͼ��ÿ�����
	int intervals;
	//ԭʼ��������
	int init_sample;
	//��ֵ�����ֵ
	float thresh;
};

