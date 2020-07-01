#pragma once
#include <vector>
#include <math.h>
using namespace std;

class Mypoint; // ����
typedef vector<Mypoint> MpVec;//�洢������
typedef vector<pair<Mypoint, Mypoint> > MpPairVec;//���ڴ洢ƥ���
class Mypoint {
public:
	//����
	~Mypoint() {};
	//����
	Mypoint() : orientation(0) {};
	//! ����-�Ի���������ռ�ľ���
	float operator-(const Mypoint &rhs)
	{
		float sum = 0.f;
		for (int i = 0; i < 64; ++i)
			sum += (this->descriptor[i] - rhs.descriptor[i])*(this->descriptor[i] - rhs.descriptor[i]);
		return sqrt(sum);
	};
   // ����
	float x, y;
	//�߶�
	float scale;
	//����
	float orientation;
	//����������
	float descriptor[64];
	//��λ��
	float dx, dy;
};
//����� ������
inline int fRound(float flt)
{
	return (int)floor(flt + 0.5f);
}

