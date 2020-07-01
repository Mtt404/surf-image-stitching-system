#include"sup.h"
using namespace std;
void drawPoint1(cv::Mat img, Mypoint &mpt)
{
	float s;
	int r1, c1;
	s = 3;
	r1 = fRound(mpt.y);
	c1 = fRound(mpt.x);
	cv::circle(img, cv::Point(c1, r1), fRound(s), cvScalar(0, 0, 255), -1);//-1����ڲ�
	cv::circle(img, cv::Point(c1, r1), fRound(s + 1), cvScalar(0, 255, 0), 2);//��������Բ�߿��

}
//ƥ�亯�� 
void getMatches(MpVec &mpts1, MpVec &mpts2, MpPairVec &matches)
{
	float dist, d1, d2;
	Mypoint *match = nullptr;
	matches.clear();
	for (unsigned int i = 0; i < mpts1.size(); i++)
	{
		d1 = d2 = FLT_MAX;//��������
		for (unsigned int j = 0; j < mpts2.size(); j++)
		{
			dist = mpts1[i] - mpts2[j];

			if (dist<d1) // ���µ���
			{
				d2 = d1;
				d1 = dist;
				match = &mpts2[j];
			}
			else if (dist<d2) // ����d2
			{
				d2 = dist;
			}
		}

		// ����ڱ�С��0.65
		if (d1 / d2 < 0.65)
		{
			// Store the change in position
			mpts1[i].dx = match->x - mpts1[i].x;
			mpts1[i].dy = match->y - mpts1[i].y;
			matches.push_back(make_pair(mpts1[i], *match));
		}
	}
}
