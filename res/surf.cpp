#include "surf.h"
const float pi = 3.14159f;
//二维高斯（sigma = 2.5）的查找表--查找更快，其中（0,0）在左上方，（6,6）在右下方 寻找主方向r=6的范围haar小波加权使用
const double gauss25[7][7] = {
	0.02546481,	0.02350698,	0.01849125,	0.01239505,	0.00708017,	0.00344629,	0.00142946,
	0.02350698,	0.02169968,	0.01706957,	0.01144208,	0.00653582,	0.00318132,	0.00131956,
	0.01849125,	0.01706957,	0.01342740,	0.00900066,	0.00514126,	0.00250252,	0.00103800,
	0.01239505,	0.01144208,	0.00900066,	0.00603332,	0.00344629,	0.00167749,	0.00069579,
	0.00708017,	0.00653582,	0.00514126,	0.00344629,	0.00196855,	0.00095820,	0.00039744,
	0.00344629,	0.00318132,	0.00250252,	0.00167749,	0.00095820,	0.00046640,	0.00019346,
	0.00142946,	0.00131956,	0.00103800,	0.00069579,	0.00039744,	0.00019346,	0.00008024
};
// 构造
Surf::Surf(cv::Mat *img, MpVec &mpts)
	: mpts(mpts)
{
	this->img = img;
}

void Surf::getDescriptors()
{
	// 检查是否有点需要被描述---即点集是否为空
	if (!mpts.size()) return;
	// 获取点集大小
	int mpts_size = (int)mpts.size();

	for (int i = 0; i < mpts_size; ++i)
	{
		index = i;
		// 先分配方向并提取旋转不变描述符
		getOrientation();
		getDescriptor();
	}

}
void Surf::getOrientation()
{
	Mypoint *mpt = &mpts[index];
	float gauss = 0.f, scale = mpt->scale;
	const int s = fRound(scale), r = fRound(mpt->y), c = fRound(mpt->x);
	std::vector<float> resX(109), resY(109), Ang(109);//x^2+y^2<=36内部共109个点
	const int id[] = { 6,5,4,3,2,1,0,1,2,3,4,5,6 };
	int idx = 0;//作计数
	//计算离中心点6S范围内的点的小波变换,每一个点的小波变换形成一个二维向量
	for (int i = -6; i <= 6; ++i)
	{
		for (int j = -6; j <= 6; ++j)
		{
			if (i*i + j*j < 36) //x^2+y^2<=36
			{
				gauss = (float)(gauss25[id[i + 6]][id[j + 6]]); //矩阵下标没有负数，故以此加回来，相当于平移 
				resX[idx] = gauss * haarX(r + j*s, c + i*s, 4 * s);//先得到Haar小波响应，再加权。
				resY[idx] = gauss * haarY(r + j*s, c + i*s, 4 * s);
				Ang[idx] = getAngle(resX[idx], resY[idx]);//计算角度,调用getAngle函数
				++idx;
			}
		}
	}
	// 计算主方向
	float sumX = 0.f, sumY = 0.f;
	float max = 0.f, orientation = 0.f;
	float ang1 = 0.f, ang2 = 0.f;
	//检测主方向,即在一个方向附近的角度范围内,上一步求出的向量在这π/3角度范围的和达到极值,这个范围的中心为主方向
	for (ang1 = 0; ang1 < 2 * pi; ang1 += 0.15f) {
		ang2 = (ang1 + pi / 3.0f > 2 * pi ? ang1 - 5.0f*pi / 3.0f : ang1 + pi / 3.0f);//超过就相当于循环回去总在0-2π
		sumX = sumY = 0.f;
		for (unsigned int k = 0; k < Ang.size(); ++k)
		{
			//从采样点的x轴起获取角度
			const float & ang = Ang[k];
			// 是否在这个窗口内（两种可能，正常模式ang1 < ang2，超出回归模式ang2 < ang1）在就求和
			if (ang1 < ang2 && ang1 < ang && ang < ang2)
			{
				sumX += resX[k];
				sumY += resY[k];
			}
			else if (ang2 < ang1 &&((ang > 0 && ang < ang2) || (ang > ang1 && ang < 2 * pi)))
			{
				sumX += resX[k];
				sumY += resY[k];
			}
		}
		if (sumX*sumX + sumY*sumY > max) //大于极值，更新极值，更新主方向
		{
			max = sumX*sumX + sumY*sumY;
			orientation = getAngle(sumX, sumY);
		}
	}
	//遍历更新完毕，赋值
	mpt->orientation = orientation;
}
void Surf::getDescriptor()
{
	int y, x, sample_x, sample_y, count = 0;
	int i = 0, ix = 0, j = 0, jx = 0, xs = 0, ys = 0;
	float scale, *desc, dx, dy, mdx, mdy, Cos, Sin;
	float gauss_s1 = 0.f, gauss_s2 = 0.f;
	float rx = 0.f, ry = 0.f, rrx = 0.f, rry = 0.f, len = 0.f;
	float cx = -0.5f, cy = 0.f; //次区域中心进行4x4高斯加权

	Mypoint *mpt = &mpts[index];
	scale = mpt->scale;
	x = fRound(mpt->x);
	y = fRound(mpt->y);
	desc = mpt->descriptor;//数组头给到指针
    //主方向在原坐标系的投影
	Cos = cos(mpt->orientation);
	Sin = sin(mpt->orientation);
	i = -8;
	//Calculate descriptor for this interest point
	while (i < 12)
	{
		j = -8;
		i = i - 4;//
		cx += 1.f;
		cy = -0.5f;
		while (j < 12)
		{
			dx = dy = mdx = mdy = 0.f;//特征描述四组值
			cy += 1.f;
			j = j - 4;
			ix = i + 5;
			jx = j + 5;
			//fRound为取最近整方法，寻找特征点旋转对应点
			xs = fRound(x + (-jx*scale*Sin + ix*scale*Cos));
			ys = fRound(y + (jx*scale*Cos + ix*scale*Sin));

			for (int k = i; k < i + 9; ++k)
			{
				for (int l = j; l < j + 9; ++l)
				{
					//获取旋转轴上采样点的坐标
					sample_x = fRound(x + (-l*scale*Sin + k*scale*Cos));
					sample_y = fRound(y + (l*scale*Cos + k*scale*Sin));

					//获取高斯权值
					gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.5f*scale);

					//获取x和yhaar响应
					rx = haarX(sample_y, sample_x, 2 * fRound(scale));
					ry = haarY(sample_y, sample_x, 2 * fRound(scale));

					//高斯加权x和y响应
					rrx = gauss_s1*(-rx*Sin + ry*Cos);
					rry = gauss_s1*(rx*Cos + ry*Sin);
					//累积和
					dx += rrx;
					dy += rry;
					mdx += fabs(rrx);
					mdy += fabs(rry);

				}
			}
			//二次高斯加权
			gauss_s2 = gaussian(cx - 2.0f, cy - 2.0f, 1.5f);
			desc[count++] = dx*gauss_s2;
			desc[count++] = dy*gauss_s2;
			desc[count++] = mdx*gauss_s2;
			desc[count++] = mdy*gauss_s2;
			len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy) * gauss_s2*gauss_s2;
			j += 9;
		}
		i += 9;
	}

	//归一化
	len = sqrt(len);
	for (int i = 0; i < 64; ++i)
		desc[i] /= len;

}
//计算x,y处 2d gaussian系数 
inline float Surf::gaussian(int x, int y, float sig)
{
	return (1.0f / (2.0f*pi*sig*sig)) * exp(-(x*x + y*y) / (2.0f*sig*sig));
}
//! 计算x,y处 2d gaussian 
inline float Surf::gaussian(float x, float y, float sig)
{
	return 1.0f / (2.0f*pi*sig*sig) * exp(-(x*x + y*y) / (2.0f*sig*sig));
}
//计算row column处小波长度为s的x方向的小波变换   箱式滤波加速
inline float Surf::haarX(int row, int column, int s)
{
	return BoxIntegral(img, row - s / 2, column, s, s / 2)- 1 * BoxIntegral(img, row - s / 2, column - s / 2, s, s / 2);
}
//! //计算row column处小波长度为s的y方向的小波变换,
inline float Surf::haarY(int row, int column, int s)
{
	return BoxIntegral(img, row, column - s / 2, s / 2, s)- 1 * BoxIntegral(img, row - s / 2, column - s / 2, s / 2, s);
}
//求角度,单位为弧度,以x轴正方向为0,逆时针,0到2pi
float Surf::getAngle(float X, float Y)
{
	if (X > 0 && Y >= 0)
		return atan(Y / X);
	if (X < 0 && Y >= 0)
		return pi - atan(-Y / X);
	if (X < 0 && Y < 0)
		return pi + atan(Y / X);
	if (X > 0 && Y < 0)
		return 2 * pi - atan(-Y / X);
	return 0;
}

