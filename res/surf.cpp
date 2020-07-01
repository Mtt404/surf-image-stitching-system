#include "surf.h"
const float pi = 3.14159f;
//��ά��˹��sigma = 2.5���Ĳ��ұ�--���Ҹ��죬���У�0,0�������Ϸ�����6,6�������·� Ѱ��������r=6�ķ�ΧhaarС����Ȩʹ��
const double gauss25[7][7] = {
	0.02546481,	0.02350698,	0.01849125,	0.01239505,	0.00708017,	0.00344629,	0.00142946,
	0.02350698,	0.02169968,	0.01706957,	0.01144208,	0.00653582,	0.00318132,	0.00131956,
	0.01849125,	0.01706957,	0.01342740,	0.00900066,	0.00514126,	0.00250252,	0.00103800,
	0.01239505,	0.01144208,	0.00900066,	0.00603332,	0.00344629,	0.00167749,	0.00069579,
	0.00708017,	0.00653582,	0.00514126,	0.00344629,	0.00196855,	0.00095820,	0.00039744,
	0.00344629,	0.00318132,	0.00250252,	0.00167749,	0.00095820,	0.00046640,	0.00019346,
	0.00142946,	0.00131956,	0.00103800,	0.00069579,	0.00039744,	0.00019346,	0.00008024
};
// ����
Surf::Surf(cv::Mat *img, MpVec &mpts)
	: mpts(mpts)
{
	this->img = img;
}

void Surf::getDescriptors()
{
	// ����Ƿ��е���Ҫ������---���㼯�Ƿ�Ϊ��
	if (!mpts.size()) return;
	// ��ȡ�㼯��С
	int mpts_size = (int)mpts.size();

	for (int i = 0; i < mpts_size; ++i)
	{
		index = i;
		// �ȷ��䷽����ȡ��ת����������
		getOrientation();
		getDescriptor();
	}

}
void Surf::getOrientation()
{
	Mypoint *mpt = &mpts[index];
	float gauss = 0.f, scale = mpt->scale;
	const int s = fRound(scale), r = fRound(mpt->y), c = fRound(mpt->x);
	std::vector<float> resX(109), resY(109), Ang(109);//x^2+y^2<=36�ڲ���109����
	const int id[] = { 6,5,4,3,2,1,0,1,2,3,4,5,6 };
	int idx = 0;//������
	//���������ĵ�6S��Χ�ڵĵ��С���任,ÿһ�����С���任�γ�һ����ά����
	for (int i = -6; i <= 6; ++i)
	{
		for (int j = -6; j <= 6; ++j)
		{
			if (i*i + j*j < 36) //x^2+y^2<=36
			{
				gauss = (float)(gauss25[id[i + 6]][id[j + 6]]); //�����±�û�и��������Դ˼ӻ������൱��ƽ�� 
				resX[idx] = gauss * haarX(r + j*s, c + i*s, 4 * s);//�ȵõ�HaarС����Ӧ���ټ�Ȩ��
				resY[idx] = gauss * haarY(r + j*s, c + i*s, 4 * s);
				Ang[idx] = getAngle(resX[idx], resY[idx]);//����Ƕ�,����getAngle����
				++idx;
			}
		}
	}
	// ����������
	float sumX = 0.f, sumY = 0.f;
	float max = 0.f, orientation = 0.f;
	float ang1 = 0.f, ang2 = 0.f;
	//���������,����һ�����򸽽��ĽǶȷ�Χ��,��һ����������������/3�Ƕȷ�Χ�ĺʹﵽ��ֵ,�����Χ������Ϊ������
	for (ang1 = 0; ang1 < 2 * pi; ang1 += 0.15f) {
		ang2 = (ang1 + pi / 3.0f > 2 * pi ? ang1 - 5.0f*pi / 3.0f : ang1 + pi / 3.0f);//�������൱��ѭ����ȥ����0-2��
		sumX = sumY = 0.f;
		for (unsigned int k = 0; k < Ang.size(); ++k)
		{
			//�Ӳ������x�����ȡ�Ƕ�
			const float & ang = Ang[k];
			// �Ƿ�����������ڣ����ֿ��ܣ�����ģʽang1 < ang2�������ع�ģʽang2 < ang1���ھ����
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
		if (sumX*sumX + sumY*sumY > max) //���ڼ�ֵ�����¼�ֵ������������
		{
			max = sumX*sumX + sumY*sumY;
			orientation = getAngle(sumX, sumY);
		}
	}
	//����������ϣ���ֵ
	mpt->orientation = orientation;
}
void Surf::getDescriptor()
{
	int y, x, sample_x, sample_y, count = 0;
	int i = 0, ix = 0, j = 0, jx = 0, xs = 0, ys = 0;
	float scale, *desc, dx, dy, mdx, mdy, Cos, Sin;
	float gauss_s1 = 0.f, gauss_s2 = 0.f;
	float rx = 0.f, ry = 0.f, rrx = 0.f, rry = 0.f, len = 0.f;
	float cx = -0.5f, cy = 0.f; //���������Ľ���4x4��˹��Ȩ

	Mypoint *mpt = &mpts[index];
	scale = mpt->scale;
	x = fRound(mpt->x);
	y = fRound(mpt->y);
	desc = mpt->descriptor;//����ͷ����ָ��
    //��������ԭ����ϵ��ͶӰ
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
			dx = dy = mdx = mdy = 0.f;//������������ֵ
			cy += 1.f;
			j = j - 4;
			ix = i + 5;
			jx = j + 5;
			//fRoundΪȡ�����������Ѱ����������ת��Ӧ��
			xs = fRound(x + (-jx*scale*Sin + ix*scale*Cos));
			ys = fRound(y + (jx*scale*Cos + ix*scale*Sin));

			for (int k = i; k < i + 9; ++k)
			{
				for (int l = j; l < j + 9; ++l)
				{
					//��ȡ��ת���ϲ����������
					sample_x = fRound(x + (-l*scale*Sin + k*scale*Cos));
					sample_y = fRound(y + (l*scale*Cos + k*scale*Sin));

					//��ȡ��˹Ȩֵ
					gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.5f*scale);

					//��ȡx��yhaar��Ӧ
					rx = haarX(sample_y, sample_x, 2 * fRound(scale));
					ry = haarY(sample_y, sample_x, 2 * fRound(scale));

					//��˹��Ȩx��y��Ӧ
					rrx = gauss_s1*(-rx*Sin + ry*Cos);
					rry = gauss_s1*(rx*Cos + ry*Sin);
					//�ۻ���
					dx += rrx;
					dy += rry;
					mdx += fabs(rrx);
					mdy += fabs(rry);

				}
			}
			//���θ�˹��Ȩ
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

	//��һ��
	len = sqrt(len);
	for (int i = 0; i < 64; ++i)
		desc[i] /= len;

}
//����x,y�� 2d gaussianϵ�� 
inline float Surf::gaussian(int x, int y, float sig)
{
	return (1.0f / (2.0f*pi*sig*sig)) * exp(-(x*x + y*y) / (2.0f*sig*sig));
}
//! ����x,y�� 2d gaussian 
inline float Surf::gaussian(float x, float y, float sig)
{
	return 1.0f / (2.0f*pi*sig*sig) * exp(-(x*x + y*y) / (2.0f*sig*sig));
}
//����row column��С������Ϊs��x�����С���任   ��ʽ�˲�����
inline float Surf::haarX(int row, int column, int s)
{
	return BoxIntegral(img, row - s / 2, column, s, s / 2)- 1 * BoxIntegral(img, row - s / 2, column - s / 2, s, s / 2);
}
//! //����row column��С������Ϊs��y�����С���任,
inline float Surf::haarY(int row, int column, int s)
{
	return BoxIntegral(img, row, column - s / 2, s / 2, s)- 1 * BoxIntegral(img, row - s / 2, column - s / 2, s / 2, s);
}
//��Ƕ�,��λΪ����,��x��������Ϊ0,��ʱ��,0��2pi
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

