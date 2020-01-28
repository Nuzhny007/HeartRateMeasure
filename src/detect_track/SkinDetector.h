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

    bool InitModel(std::string modelPath = "../beatmagnifier/data/");
    bool SaveModel(std::string modelPath = "../beatmagnifier/data/");
    bool LearnModel(std::string dataPath = "../beatmagnifier/data/");

    cv::Mat Detect(cv::Mat image, bool drawResults, bool saveResult, int frameInd);

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

///
/// \brief SkinInit
/// \param skinDetector
/// \return
///
inline bool SkinInit(SkinDetector& skinDetector, const std::string& skinPath)
{
    bool res = skinDetector.InitModel(skinPath);

    if (!res)
    {
        res = skinDetector.LearnModel(skinPath);
        if (!res)
        {
            std::cout << "Skin detector wasn't initializad!" << std::endl;
        }
        else
        {
            skinDetector.SaveModel(skinPath);
        }
    }
    return res;
}
