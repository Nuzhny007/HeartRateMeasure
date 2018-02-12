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

    bool InitModel(std::string modelPath = "../HeartRateMeasure/data/");
    bool SaveModel(std::string modelPath = "../HeartRateMeasure/data/");
    bool LearnModel(std::string dataPath = "../HeartRateMeasure/data/");

    cv::Mat Detect(cv::Mat image, bool saveResult, int frameInd);

private:
    cv::Mat m_skinMask;

    std::string m_dataFileName;
    std::string m_modelFileName;

    bool m_useRGB;

    ///
    /// \brief m_model
    /// Skin Classifier
    ///
    cv::Ptr<cv::ml::StatModel> m_model;
};
