#include "SimpleMA.h"

///
SimpleMA::SimpleMA()
{
    m_first = true;
    m_blurredSize = cv::Size(10, 10);
	m_fLow = 70 / 60.f / 10;
	m_fHigh = 80 / 60.f / 10;
    m_alpha = 200;
}

///
SimpleMA::~SimpleMA()
{
	Release();
}

///
bool SimpleMA::IsInitialized() const
{
	return !m_first;
}

///
cv::Size SimpleMA::GetSize() const
{
	return m_srcFloat.size();
}

///
void SimpleMA::Init(const cv::UMat& rgbframe, int /*alpha*/, int /*lambda_c*/, float /*fl*/, float /*fh*/, int /*samplingRate*/, float /*chromAttenuation*/)
{
	Release();

	//m_blurredSize.width = rgbframe.cols / 8;
	//m_blurredSize.height = rgbframe.rows / 8;

	// convert to float
	rgbframe.convertTo(m_srcFloat, CV_32F);

	// apply spatial filter: blur and downsample
	cv::resize(m_srcFloat, m_blurred, m_blurredSize, 0, 0, cv::INTER_AREA);

	m_first = false;
	m_blurred.copyTo(m_lowpassHigh);
	m_blurred.copyTo(m_lowpassLow);
}

///
void SimpleMA::Release()
{
	m_first = true;
}

///
cv::UMat SimpleMA::Process(const cv::UMat& rgbframe)
{
	// convert to float
	rgbframe.convertTo(m_srcFloat, CV_32FC3);

	// apply spatial filter: blur and downsample
	resize(m_srcFloat, m_blurred, m_blurredSize, 0, 0, cv::INTER_AREA);

	// apply temporal filter: subtraction of two IIR lowpass filters
	m_lowpassHigh = (1 - m_fHigh) * m_lowpassHigh + m_fHigh * m_blurred;
	m_lowpassLow = (1 - m_fLow) * m_lowpassLow + m_fLow * m_blurred;
	m_blurred = m_alpha * (m_lowpassHigh - m_lowpassLow);

	// resize back to original size
	resize(m_blurred, m_outFloat, rgbframe.size(), 0, 0, cv::INTER_LINEAR);

	// add back to original frame
	m_outFloat += m_srcFloat;

	return m_outFloat.getUMat(cv::ACCESS_READ);
}
