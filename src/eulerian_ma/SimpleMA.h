#pragma once

#include "MotionAmp.h"

class SimpleMA : public MotionAmp
{
public:
    SimpleMA();
    ~SimpleMA();

	bool IsInitialized() const;
	cv::Size GetSize() const;

	void Init(const cv::UMat& rgbframe, int alpha, int lambda_c, float fl, float fh, int samplingRate, float chromAttenuation);
	void Release();
	cv::UMat Process(const cv::UMat& rgbframe);

private:
	cv::Mat m_srcFloat;
	cv::Mat m_blurred;
	cv::Mat m_lowpassHigh;
	cv::Mat m_lowpassLow;
	cv::Mat m_outFloat;

	bool m_first;
	cv::Size m_blurredSize;
	float m_fLow;
	float m_fHigh;
	int m_alpha;
};
