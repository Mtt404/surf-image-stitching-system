#include "integral.h"
#include "Mypoint.h"
//include "utils.h"
#include <vector>
#include "responselayer.h"
#include "fasthessian.h"
using namespace std;
//构造函数
FastHessian::FastHessian(cv::Mat *img, std::vector<Mypoint> &mpts,
	const int octaves, const int intervals, const int init_sample,
	const float thresh)
	: mpts(mpts), i_width(0), i_height(0)
{
	// 给就设置，不给就默认
	saveParameters(octaves, intervals, init_sample, thresh);
	setIntImage(img);
}
//析构
FastHessian::~FastHessian()
{
	for (unsigned int i = 0; i < responseMap.size(); ++i)
	{
		delete responseMap[i];
	}
}
void FastHessian::saveParameters(const int octaves, const int intervals,const int init_sample, const float thresh)
{
	// 参数设置并检查，符合即可，不符合默认
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
	// 参数保存
	this->img = img;
	i_height = img->rows;
	i_width = img->cols;
}
//得到特征点
void FastHessian::getMypoints(int flag)
{
	// 高斯滤波核的索引
	static const int filter_map[OCTAVES][INTERVALS] = { { 0,1,2,3 },{ 1,3,4,5 },{ 3,5,6,7 },{ 5,7,8,9 },{ 7,9,10,11 } };
	// 清空备用
	mpts.clear();
	// 建立相应图
	buildResponseMap();

	// 构建响应层
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
//构建尺度空间下的响应图
void FastHessian::buildResponseMap()
{
	// 高斯滤波核前四组尺寸大小
	// Oct1: 9,  15, 21, 27
	// Oct2: 15, 27, 39, 51
	// Oct3: 27, 51, 75, 99
	// Oct4: 51, 99, 147,195
	// 清除已经存在的层备用
	for (unsigned int i = 0; i < responseMap.size(); ++i)
		delete responseMap[i];
	responseMap.clear();

	// 得到图像的参数
	int w = (i_width / init_sample);//宽=原始图像宽/原始抽样倍数
	int h = (i_height / init_sample);//高=原始图像高/原始抽样倍数
	int s = (init_sample);//原始抽样倍数

    // 创建尺度空间所有层
	if (octaves >= 1)
	{
		responseMap.push_back(new ResponseLayer(w, h, s, 9));
		responseMap.push_back(new ResponseLayer(w, h, s, 15));
		responseMap.push_back(new ResponseLayer(w, h, s, 21));
		responseMap.push_back(new ResponseLayer(w, h, s, 27));
	}
	if (octaves >= 2)
	{
		responseMap.push_back(new ResponseLayer(w / 2, h / 2, s * 2, 39));//尺度变大，相对图像变小
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
	// 提取每一层的hessian值及laplacian值
	for (unsigned int i = 0; i < responseMap.size(); ++i)
	{
		buildResponseLayer(responseMap[i]);
	}
}
//!计算尺度空间每层的hessian值
void FastHessian::buildResponseLayer(ResponseLayer *res)
{
	float *responses = res->responses;          //hessian值存储数组(指针)
	int step = res->step;                      // 步长
	int b = (res->filter - 1) / 2;             // 减掉中心一像素，除以2高斯滤波核边界
	int l = res->filter / 3;                   // (filter size / 3)滤波器的波瓣
	int w = res->filter;                       // 高斯滤波核的大小
	float inverse_area = 1.f / (w*w);           // 归一化因子
	float Dxx, Dyy, Dxy;
	//计算每个像素点的hessian值及laplacian值（即迹的值)
	for (int r, c, ar = 0, index = 0; ar < res->height; ++ar)
	{
		for (int ac = 0; ac < res->width; ++ac, index++)
		{
			//得到像素在对应图像中的坐标位置
			r = ar * step;
			c = ac * step;

			// 计算hessian成员值
			Dxx = BoxIntegral(img, r - l + 1, c - b, 2 * l - 1, w)
				- BoxIntegral(img, r - l + 1, c - l / 2, 2 * l - 1, l) * 3;
			Dyy = BoxIntegral(img, r - b, c - l + 1, w, 2 * l - 1)//w滤波器大小，如9,2*l-1=5，实际就是9*5的的一个区域
				- BoxIntegral(img, r - l / 2, c - l + 1, l, 2 * l - 1) * 3;
			Dxy = (-BoxIntegral(img, r - l, c + 1, l, l))
				- BoxIntegral(img, r + 1, c - l, l, l)
				+ BoxIntegral(img, r - l, c - l, l, l)
				+ BoxIntegral(img, r + 1, c + 1, l, l);

			// 归一化
			Dxx *= inverse_area;
			Dyy *= inverse_area;
			Dxy *= inverse_area;

			// 保存
			responses[index] = (Dxx * Dyy - 0.81f * Dxy * Dxy);//0.9*0.9=0.81//行列式值
		}
	}
}
// 极值点检测，r，c像素点坐标，t，m，b，尺度空间中的三层
int FastHessian::isExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b)
{
	//计算边界，太靠边界舍去
	int layerBorder = (t->filter + 1) / (2 * t->step);//filter通常是奇数，加一为偶数
	if (r <= layerBorder || r >= t->height - layerBorder || c <= layerBorder || c >= t->width - layerBorder)
		return 0;
	// 检测hessian值是否大于阀值，如果小于，返回0，舍去
	float candidate = m->getResponse(r, c, t);
	if (candidate < thresh)
		return 0;
	// 与附近26像素比较3*3*3=27
	for (int rr = -1; rr <= 1; ++rr)
	{
		for (int cc = -1; cc <= 1; ++cc)
		{
			//在3*3*3的邻域范围内判断它是否是极大值
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
	//计算边界，太靠边界舍去
	int layerBorder = (t->filter + 1) / (2 * t->step);//filter通常是奇数，加一为偶数
	if (r <= layerBorder || r >= t->height - layerBorder || c <= layerBorder || c >= t->width - layerBorder)
		return 0;
	//左图左半部分，舍去
	
	if (c < (i_width / (2 * t->step)))
		return 0;
	// 检测hessian值是否大于阀值，如果小于，返回0，舍去
	float candidate = m->getResponse(r, c, t);
	if (candidate < thresh)
		return 0;
	// 与附近26像素比较3*3*3=27
	for (int rr = -1; rr <= 1; ++rr)
	{
		for (int cc = -1; cc <= 1; ++cc)
		{
			//在3*3*3的邻域范围内判断它是否是极大值
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
	//计算边界，太靠边界舍去
	int layerBorder = (t->filter + 1) / (2 * t->step);//filter通常是奇数，加一为偶数
	if (r <= layerBorder || r >= t->height - layerBorder || c <= layerBorder || c >= t->width - layerBorder)
		return 0;
	//右图右半部分，舍去
	if (c > (i_width/(2*t->step)))
		return 0;
	// 检测hessian值是否大于阀值，如果小于，返回0，舍去
	float candidate = m->getResponse(r, c, t);
	if (candidate < thresh)
		return 0;
	// 与附近26像素比较3*3*3=27
	for (int rr = -1; rr <= 1; ++rr)
	{
		for (int cc = -1; cc <= 1; ++cc)
		{
			//在3*3*3的邻域范围内判断它是否是极大值
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
//插值法逼近极值点准确位置，r、c像素点坐标，t，m，b尺度空间连续三层
void FastHessian::interpolateExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b)
{
	//插值计算准确的极值
	//确保中间层
	int filterStep = (m->filter - b->filter);
	assert(filterStep > 0 && t->filter - m->filter == m->filter - b->filter);

	//调用插值函数
	double xi = 0, xr = 0, xc = 0;
	interpolateStep(r, c, t, m, b, &xi, &xr, &xc);

	//如果偏移没有超出0.5,的范围
	if (fabs(xi) < 0.5f  &&  fabs(xr) < 0.5f  &&  fabs(xc) < 0.5f)
	{
		Mypoint Mpt;
		Mpt.x = (float)((c + xc) * t->step);
		Mpt.y = (float)((r + xr) * t->step);
		Mpt.scale = (float)((0.1333f) * (m->filter + xi * filterStep));//1.2/9=0.1333333 + xi * filterStep
		mpts.push_back(Mpt);
	}
}
//! 插值法算出极值点偏差
void FastHessian::interpolateStep(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b,double* xi, double* xr, double* xc)
{
	cv::Mat dD, H;
	double x[3] = { 0 };
	dD = deriv3D(r, c, t, m, b);//三维偏导数计算
	H = hessian3D(r, c, t, m, b);//三维hessian矩阵计算 二阶导
	cv::Mat H_inv(3, 3, CV_64FC1);//创建64位单精度3*3矩阵
	cv::invert(H, H_inv);//求H的逆矩阵,伪逆矩阵
	cv::Mat X(3, 1, CV_64FC1);
	int step = X.step / sizeof(double);
	//公式
	cv::gemm(H_inv, dD, -1, NULL, 0, X, 0);//
	double* pData = (double*)X.data;
	x[2] = pData[2*step];
	x[1] = pData[step];
	x[0] = pData[0];
	*xi = x[2];//三个方向偏差
	*xr = x[1];
	*xc = x[0];
}
// 求三维偏导数
cv::Mat FastHessian::deriv3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b)
{
	cv::Mat dH(3, 1, CV_64FC1);
	double dx, dy, ds;
	dx = (m->getResponse(r, c + 1, t) - m->getResponse(r, c - 1, t)) / 2.0;//用中心差分代替
	dy = (m->getResponse(r + 1, c, t) - m->getResponse(r - 1, c, t)) / 2.0;
	ds = (t->getResponse(r, c) - b->getResponse(r, c, t)) / 2.0;
	int step = dH.step / sizeof(double);
	double* pData = (double*)dH.data;
	pData[0] = dx;//赋值
	pData[step] = dy;
	pData[2*step] = ds;
	return dH;
}
//三维hessian矩阵计算
cv::Mat FastHessian::hessian3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b)
{
	cv::Mat H(3, 3, CV_64FC1);
	double v, dxx, dyy, dss, dxy, dxs, dys;
	v = m->getResponse(r, c, t);
	//二维函数f(x,y)中，x,y两个方向的二阶差分
	dxx = m->getResponse(r, c + 1, t) + m->getResponse(r, c - 1, t) - 2 * v;//x方向--列之间
	dyy = m->getResponse(r + 1, c, t) + m->getResponse(r - 1, c, t) - 2 * v;
	dss = t->getResponse(r, c) + b->getResponse(r, c, t) - 2 * v;
	//二阶混合导根据一阶中心差分可推，也可泰勒展开
	dxy = (m->getResponse(r + 1, c + 1, t) - m->getResponse(r + 1, c - 1, t) -
		m->getResponse(r - 1, c + 1, t) + m->getResponse(r - 1, c - 1, t)) / 4.0;//
	dxs = (t->getResponse(r, c + 1) - t->getResponse(r, c - 1) -
		b->getResponse(r, c + 1, t) + b->getResponse(r, c - 1, t)) / 4.0;
	dys = (t->getResponse(r + 1, c) - t->getResponse(r - 1, c) -
		b->getResponse(r + 1, c, t) + b->getResponse(r - 1, c, t)) / 4.0;
	int step = H.step/sizeof(double);
	double* pData = (double*)H.data;
	pData[0] = dxx;//赋值
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



