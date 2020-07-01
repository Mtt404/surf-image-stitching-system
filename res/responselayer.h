#pragma once
#include <memory.h>
class ResponseLayer
{
public:
	int width, height, step, filter;
	float *responses;//
	//构造
	ResponseLayer(int width, int height, int step, int filter)
	{
		this->width = width;
		this->height = height;
		this->step = step;
		this->filter = filter;
		responses = new float[width*height];
		memset(responses, 0, sizeof(float)*width*height);//初始化
	}
	~ResponseLayer()
	{
		if (responses) delete[] responses;
	}
	inline float getResponse(unsigned int row, unsigned int column)
	{
		return responses[row * width + column];
	}
	inline float getResponse(unsigned int row, unsigned int column, ResponseLayer *src)
	{
		int scale = this->width / src->width;
		return responses[(scale * row) * width + (scale * column)];
	}
};

