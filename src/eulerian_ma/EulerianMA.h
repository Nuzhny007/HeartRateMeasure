#pragma once

#include "MotionAmp.h"

///
/// \brief The EulerianMA class
///
class EulerianMA : public MotionAmp
{
public:
    EulerianMA();
    ~EulerianMA();

    bool IsInitialized() const;
	cv::Size GetSize() const;

    void Init(const cv::UMat& rgbframe, int alpha, int lambda_c, float fl, float fh, int samplingRate, float chromAttenuation);
    void Release();
    cv::UMat Process(const cv::UMat& rgbframe);

private:
    std::vector<cv::Mat> m_pyr;
    std::vector<cv::Mat> m_lowpass1;
    std::vector<cv::Mat> m_lowpass2;
    std::vector<cv::Mat> m_pyrPrev;
    std::vector<cv::Size> m_pind;

    float m_chromAttenuation;
    int m_alpha;
    int m_lambda_c;
    float m_delta;
    float m_exaggeration_factor;

    float low_a[2];
    float* low_b;
    float high_a[2];
    float* high_b;

	std::vector<cv::Mat> m_filtered;
	cv::Mat m_ntscFrame;
	cv::Mat m_output;

	void lappyr(cv::Mat& src, int level, std::vector<cv::Mat>& lap_arr, std::vector<cv::Size>& nind);
	cv::Mat m_tmpDown;
	cv::Mat m_tmpUp;
	cv::Mat m_tmpDst;

	void TemporalFilter(std::vector<cv::Mat>& lowPass, const float* coeff_a, const float* coeff_b);
};
