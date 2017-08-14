#pragma once

#include "opencv2/opencv.hpp"

///
/// \brief The SkinDetector class
///
class SkinDetector
{
public:
    SkinDetector();
    ~SkinDetector();

    bool Init(std::string modelPath = "../HeartRateMeasure/data/");
    bool Learn(std::string dataPath = "../HeartRateMeasure/data/");

    cv::Mat Detect(cv::Mat image);

private:
    cv::Mat m_skinMask;
};
