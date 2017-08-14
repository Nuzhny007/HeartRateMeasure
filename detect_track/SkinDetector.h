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

    std::string m_dataFileName;
    std::string m_modelFileName;

    bool m_useRGB;

    ///
    /// \brief m_dtree
    /// DecisionTree Classifier
    ///
    cv::Ptr<cv::ml::StatModel> m_dtree;
};
