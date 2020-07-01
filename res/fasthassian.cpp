#include "integral.h"
#include "Mypoint.h"
//include "utils.h"
#include <vector>
#include "responselayer.h"
#include "fasthessian.h"
using namespace std;
//���캯��
FastHessian::FastHessian(cv::Mat *img, std::vector<Mypoint> &mpts,
	const int octaves, const int intervals, const int init_sample,
	const float thresh)
	: mpts(mpts), i_width(0), i_height(0)
{
	// �������ã�������Ĭ��
	saveParameters(octaves, intervals, init_sample, thresh);
	setIntImage(img);
}
//����
FastHessian::~FastHessian()
{
	for (unsigned int i = 0; i < responseMap.size(); ++i)
	{
		delete responseMap[i];
	}
}
void FastHessian::saveParameters(const int octaves, const int intervals,const int init_sample, const float thresh)
{
	// �������ò���飬���ϼ��ɣ�������Ĭ��
	this->octaves =
		(octaves > 0 && octaves <= 4 ? octaves : OCTAVES);
	this->intervals =
		(intervals > 0 && intervals <= 4 ? intervals : INTERVALS);
	this->init_sample =
		(init_sample > 0 && init_sample <= 6 ? init_sample : INIT_SAMPLE);
	this->thresh = (thresh >= 0 ? thresh : THRES);
}
void FastHessian::setIntImage(cv::Mat *img)
{
	// ��������
	this->img = img;
	i_height = img->rows;
	i_width = img->cols;
}
//�õ�������
void FastHessian::getMypoints(int flag)
{
	// ��˹�˲��˵�����
	static const int filter_map[OCTAVES][INTERVALS] = { { 0,1,2,3 },{ 1,3,4,5 },{ 3,5,6,7 },{ 5,7,8,9 },{ 7,9,10,11 } };
	// ��ձ���
	mpts.clear();
	// ������Ӧͼ
	buildResponseMap();

	// ������Ӧ��
	ResponseLayer *b, *m, *t;//b=bottom m=middle t=top
	for (int o = 0; o < octaves; ++o) for (int i = 0; i <= 1; ++i)
	{
		b = responseMap.at(filter_map[o][i]);
		m = responseMap.at(filter_map[o][i + 1]);
		t = responseMap.at(filter_map[o][i + 2]);

		for (int r = 0; r < t->height; ++r)
		{
			for (int c = 0; c < t->width; ++c)
			{
				if (flag == 1)
				{
					if (isExtremum1(r, c, t, m, b))
					{
						interpolateExtremum(r, c, t, m, b);
					}
				}
				else if (flag == 2)
				{
					if (isExtremum2(r, c, t, m, b))
					{
						interpolateExtremum(r, c, t, m, b);
					}
				}
				else
				{
					if (isExtremum(r, c, t, m, b))
					{
						interpolateExtremum(r, c, t, m, b);
					}
				}
			}
		}
	}
}
//�����߶ȿռ��µ���Ӧͼ
void FastHessian::buildResponseMap()
{
	// ��˹�˲���ǰ����ߴ��С
	// Oct1: 9,  15, 21, 27
	// Oct2: 15, 27, 39, 51
	// Oct3: 27, 51, 75, 99
	// Oct4: 51, 99, 147,195
	// ����Ѿ����ڵĲ㱸��
	for (unsigned int i = 0; i < responseMap.size(); ++i)
		delete responseMap[i];
	responseMap.clear();

	// �õ�ͼ��Ĳ���
	int w = (i_width / init_sample);//��=ԭʼͼ���/ԭʼ��������
	int h = (i_height / init_sample);//��=ԭʼͼ���/ԭʼ��������
	int s = (init_sample);//ԭʼ��������

    // �����߶ȿռ����в�
	if (octaves >= 1)
	{
		responseMap.push_back(new ResponseLayer(w, h, s, 9));
		responseMap.push_back(new ResponseLayer(w, h, s, 15));
		responseMap.push_back(new ResponseLayer(w, h, s, 21));
		responseMap.push_back(new ResponseLayer(w, h, s, 27));
	}
	if (octaves >= 2)
	{
		responseMap.push_back(new ResponseLayer(w / 2, h / 2, s * 2, 39));//�߶ȱ�����ͼ���С
		responseMap.push_back(new ResponseLayer(w / 2, h / 2, s * 2, 51));
	}
	if (octaves >= 3)
	{
		responseMap.push_back(new ResponseLayer(w / 4, h / 4, s * 4, 75));
		responseMap.push_back(new ResponseLayer(w / 4, h / 4, s * 4, 99));
	}
	if (octaves >= 4)
	{
		responseMap.push_back(new ResponseLayer(w / 8, h / 8, s * 8, 147));
		responseMap.push_back(new ResponseLayer(w / 8, h / 8, s * 8, 195));
	}	
	// ��ȡÿһ���hessianֵ��laplacianֵ
	for (unsigned int i = 0; i < responseMap.size(); ++i)
	{
		buildResponseLayer(responseMap[i]);
	}
}
//!����߶ȿռ�ÿ���hessianֵ
void FastHessian::buildResponseLayer(ResponseLayer *res)
{
	float *responses = res->responses;          //hessianֵ�洢����(ָ��)
	int step = res->step;                      // ����
	int b = (res->filter - 1) / 2;             // ��������һ���أ�����2��˹�˲��˱߽�
	int l = res->filter / 3;                   // (filter size / 3)�˲����Ĳ���
	int w = res->filter;                       // ��˹�˲��˵Ĵ�С
	float inverse_area = 1.f / (w*w);           // ��һ������
	float Dxx, Dyy, Dxy;
	//����ÿ�����ص��hessianֵ��laplacianֵ��������ֵ)
	for (int r, c, ar = 0, index = 0; ar < res->height; ++ar)
	{
		for (int ac = 0; ac < res->width; ++ac, index++)
		{
			//�õ������ڶ�Ӧͼ���е�����λ��
			r = ar * step;
			c = ac * step;

			// ����hessian��Աֵ
			Dxx = BoxIntegral(img, r - l + 1, c - b, 2 * l - 1, w)
				- BoxIntegral(img, r - l + 1, c - l / 2, 2 * l - 1, l) * 3;
			Dyy = BoxIntegral(img, r - b, c - l + 1, w, 2 * l - 1)//w�˲�����С����9,2*l-1=5��ʵ�ʾ���9*5�ĵ�һ������
				- BoxIntegral(img, r - l / 2, c - l + 1, l, 2 * l - 1) * 3;
			Dxy = (-BoxIntegral(img, r - l, c + 1, l, l))
				- BoxIntegral(img, r + 1, c - l, l, l)
				+ BoxIntegral(img, r - l, c - l, l, l)
				+ BoxIntegral(img, r + 1, c + 1, l, l);

			// ��һ��
			Dxx *= inverse_area;
			Dyy *= inverse_area;
			Dxy *= inverse_area;

			// ����
			responses[index] = (Dxx * Dyy - 0.81f * Dxy * Dxy);//0.9*0.9=0.81//����ʽֵ
		}
	}
}
// ��ֵ���⣬r��c���ص����꣬t��m��b���߶ȿռ��е�����
int FastHessian::isExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b)
{
	//����߽磬̫���߽���ȥ
	int layerBorder = (t->filter + 1) / (2 * t->step);//filterͨ������������һΪż��
	if (r <= layerBorder || r >= t->height - layerBorder || c <= layerBorder || c >= t->width - layerBorder)
		return 0;
	// ���hessianֵ�Ƿ���ڷ�ֵ�����С�ڣ�����0����ȥ
	float candidate = m->getResponse(r, c, t);
	if (candidate < thresh)
		return 0;
	// �븽��26���رȽ�3*3*3=27
	for (int rr = -1; rr <= 1; ++rr)
	{
		for (int cc = -1; cc <= 1; ++cc)
		{
			//��3*3*3������Χ���ж����Ƿ��Ǽ���ֵ
			if (
				t->getResponse(r + rr, c + cc) >= candidate ||
				((rr != 0 || cc != 0) && m->getResponse(r + rr, c + cc, t) >= candidate) ||
				b->getResponse(r + rr, c + cc, t) >= candidate
				)
				return 0;
		}
	}
	return 1;
}
int FastHessian::isExtremum1(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b)
{
	//����߽磬̫���߽���ȥ
	int layerBorder = (t->filter + 1) / (2 * t->step);//filterͨ������������һΪż��
	if (r <= layerBorder || r >= t->height - layerBorder || c <= layerBorder || c >= t->width - layerBorder)
		return 0;
	//��ͼ��벿�֣���ȥ
	
	if (c < (i_width / (2 * t->step)))
		return 0;
	// ���hessianֵ�Ƿ���ڷ�ֵ�����С�ڣ�����0����ȥ
	float candidate = m->getResponse(r, c, t);
	if (candidate < thresh)
		return 0;
	// �븽��26���رȽ�3*3*3=27
	for (int rr = -1; rr <= 1; ++rr)
	{
		for (int cc = -1; cc <= 1; ++cc)
		{
			//��3*3*3������Χ���ж����Ƿ��Ǽ���ֵ
			if (
				t->getResponse(r + rr, c + cc) >= candidate ||
				((rr != 0 || cc != 0) && m->getResponse(r + rr, c + cc, t) >= candidate) ||
				b->getResponse(r + rr, c + cc, t) >= candidate
				)
				return 0;
		}
	}
	return 1;
}
int FastHessian::isExtremum2(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b)
{
	//����߽磬̫���߽���ȥ
	int layerBorder = (t->filter + 1) / (2 * t->step);//filterͨ������������һΪż��
	if (r <= layerBorder || r >= t->height - layerBorder || c <= layerBorder || c >= t->width - layerBorder)
		return 0;
	//��ͼ�Ұ벿�֣���ȥ
	if (c > (i_width/(2*t->step)))
		return 0;
	// ���hessianֵ�Ƿ���ڷ�ֵ�����С�ڣ�����0����ȥ
	float candidate = m->getResponse(r, c, t);
	if (candidate < thresh)
		return 0;
	// �븽��26���رȽ�3*3*3=27
	for (int rr = -1; rr <= 1; ++rr)
	{
		for (int cc = -1; cc <= 1; ++cc)
		{
			//��3*3*3������Χ���ж����Ƿ��Ǽ���ֵ
			if (
				t->getResponse(r + rr, c + cc) >= candidate ||
				((rr != 0 || cc != 0) && m->getResponse(r + rr, c + cc, t) >= candidate) ||
				b->getResponse(r + rr, c + cc, t) >= candidate
				)
				return 0;
		}
	}
	return 1;
}
//��ֵ���ƽ���ֵ��׼ȷλ�ã�r��c���ص����꣬t��m��b�߶ȿռ���������
void FastHessian::interpolateExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b)
{
	//��ֵ����׼ȷ�ļ�ֵ
	//ȷ���м��
	int filterStep = (m->filter - b->filter);
	assert(filterStep > 0 && t->filter - m->filter == m->filter - b->filter);

	//���ò�ֵ����
	double xi = 0, xr = 0, xc = 0;
	interpolateStep(r, c, t, m, b, &xi, &xr, &xc);

	//���ƫ��û�г���0.5,�ķ�Χ
	if (fabs(xi) < 0.5f  &&  fabs(xr) < 0.5f  &&  fabs(xc) < 0.5f)
	{
		Mypoint Mpt;
		Mpt.x = (float)((c + xc) * t->step);
		Mpt.y = (float)((r + xr) * t->step);
		Mpt.scale = (float)((0.1333f) * (m->filter + xi * filterStep));//1.2/9=0.1333333 + xi * filterStep
		mpts.push_back(Mpt);
	}
}
//! ��ֵ�������ֵ��ƫ��
void FastHessian::interpolateStep(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b,double* xi, double* xr, double* xc)
{
	cv::Mat dD, H;
	double x[3] = { 0 };
	dD = deriv3D(r, c, t, m, b);//��άƫ��������
	H = hessian3D(r, c, t, m, b);//��άhessian������� ���׵�
	cv::Mat H_inv(3, 3, CV_64FC1);//����64λ������3*3����
	cv::invert(H, H_inv);//��H�������,α�����
	cv::Mat X(3, 1, CV_64FC1);
	int step = X.step / sizeof(double);
	//��ʽ
	cv::gemm(H_inv, dD, -1, NULL, 0, X, 0);//
	double* pData = (double*)X.data;
	x[2] = pData[2*step];
	x[1] = pData[step];
	x[0] = pData[0];
	*xi = x[2];//��������ƫ��
	*xr = x[1];
	*xc = x[0];
}
// ����άƫ����
cv::Mat FastHessian::deriv3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b)
{
	cv::Mat dH(3, 1, CV_64FC1);
	double dx, dy, ds;
	dx = (m->getResponse(r, c + 1, t) - m->getResponse(r, c - 1, t)) / 2.0;//�����Ĳ�ִ���
	dy = (m->getResponse(r + 1, c, t) - m->getResponse(r - 1, c, t)) / 2.0;
	ds = (t->getResponse(r, c) - b->getResponse(r, c, t)) / 2.0;
	int step = dH.step / sizeof(double);
	double* pData = (double*)dH.data;
	pData[0] = dx;//��ֵ
	pData[step] = dy;
	pData[2*step] = ds;
	return dH;
}
//��άhessian�������
cv::Mat FastHessian::hessian3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b)
{
	cv::Mat H(3, 3, CV_64FC1);
	double v, dxx, dyy, dss, dxy, dxs, dys;
	v = m->getResponse(r, c, t);
	//��ά����f(x,y)�У�x,y��������Ķ��ײ��
	dxx = m->getResponse(r, c + 1, t) + m->getResponse(r, c - 1, t) - 2 * v;//x����--��֮��
	dyy = m->getResponse(r + 1, c, t) + m->getResponse(r - 1, c, t) - 2 * v;
	dss = t->getResponse(r, c) + b->getResponse(r, c, t) - 2 * v;
	//���׻�ϵ�����һ�����Ĳ�ֿ��ƣ�Ҳ��̩��չ��
	dxy = (m->getResponse(r + 1, c + 1, t) - m->getResponse(r + 1, c - 1, t) -
		m->getResponse(r - 1, c + 1, t) + m->getResponse(r - 1, c - 1, t)) / 4.0;//
	dxs = (t->getResponse(r, c + 1) - t->getResponse(r, c - 1) -
		b->getResponse(r, c + 1, t) + b->getResponse(r, c - 1, t)) / 4.0;
	dys = (t->getResponse(r + 1, c) - t->getResponse(r - 1, c) -
		b->getResponse(r + 1, c, t) + b->getResponse(r - 1, c, t)) / 4.0;
	int step = H.step/sizeof(double);
	double* pData = (double*)H.data;
	pData[0] = dxx;//��ֵ
	pData[1] = dxy;
	pData[2] = dxs;
	pData[step + 0] = dxy;
	pData[step + 1] = dyy;
	pData[step + 2] = dys;
	pData[2 * step + 0] = dxs;
	pData[2 * step + 1] = dys;
	pData[2 * step + 2] = dss;
	return H;
}



