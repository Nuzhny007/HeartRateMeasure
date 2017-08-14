#include "SkinDetector.h"

///
/// \brief SkinDetector::SkinDetector
///
SkinDetector::SkinDetector()
{

}

///
/// \brief SkinDetector::~SkinDetector
///
SkinDetector::~SkinDetector()
{

}

///
/// \brief SkinDetector::Detect
/// \param image
/// \return
///
cv::Mat SkinDetector::Detect(cv::Mat image)
{
    return m_skinMask;
}

///
/// \brief SkinDetector::Init
/// \param modelPath
/// \return
///
bool SkinDetector::Init(std::string modelPath)
{
    return false;
}

///
/// \brief SkinDetector::Learn
/// \param dataPath
/// \return
///
bool SkinDetector::Learn(std::string dataPath)
{
    return false;
}
