#pragma once

#include <opencv2/opencv.hpp>

///
/// \brief The EulerianMA class
///
class EulerianMA
{
public:
    EulerianMA();
    ~EulerianMA();

    void Init(cv::Mat rgbframe, int alpha, int lambda_c, float fl, float fh, int samplingRate,float chromAttenuation);
    void Release();
    cv::Mat Process(cv::Mat rgbframe);

private:
    std::vector<cv::Mat> pyr;
    std::vector<cv::Mat> lowpass1;
    std::vector<cv::Mat> lowpass2;
    std::vector<cv::Mat> pyr_prev;
    std::vector<cv::Size> pind;

    float m_chromAttenuation;
    int m_alpha;
    int m_lambda_c;
    double m_delta;
    double m_exaggeration_factor;

    double low_a[2];
    double* low_b;
    double high_a[2];
    double* high_b;
};
