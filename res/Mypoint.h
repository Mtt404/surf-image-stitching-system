#pragma once
#include <vector>
#include <math.h>
using namespace std;

class Mypoint; // 声明
typedef vector<Mypoint> MpVec;//存储特征点
typedef vector<pair<Mypoint, Mypoint> > MpPairVec;//用于存储匹配对
class Mypoint {
public:
	//析构
	~Mypoint() {};
	//构造
	Mypoint() : orientation(0) {};
	//! 重载-以获得在描述空间的距离
	float operator-(const Mypoint &rhs)
	{
		float sum = 0.f;
		for (int i = 0; i < 64; ++i)
			sum += (this->descriptor[i] - rhs.descriptor[i])*(this->descriptor[i] - rhs.descriptor[i]);
		return sqrt(sum);
	};
   // 坐标
	float x, y;
	//尺度
	float scale;
	//方向
	float orientation;
	//描述子向量
	float descriptor[64];
	//点位移
	float dx, dy;
};
//最近邻 用两次
inline int fRound(float flt)
{
	return (int)floor(flt + 0.5f);
}

