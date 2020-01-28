#pragma once
#include <opencv2/opencv.hpp>

///
class MotionAmp
{
public:
	MotionAmp() = default;
	virtual ~MotionAmp()
	{

	}

	virtual bool IsInitialized() const = 0;
	virtual cv::Size GetSize() const = 0;

	virtual void Init(const cv::UMat& rgbframe,
		int alpha, int lambda_c, float fl, float fh, int samplingRate, float chromAttenuation) = 0;
	virtual void Release() = 0;
	virtual cv::UMat Process(const cv::UMat& rgbframe) = 0;
};
