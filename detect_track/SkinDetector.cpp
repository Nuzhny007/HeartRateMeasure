#include "SkinDetector.h"
#include <fstream>

///
/// \brief SkinDetector::SkinDetector
///
SkinDetector::SkinDetector()
    :
      m_dataFileName("Skin_NonSkin.txt"),
      m_modelFileName(""),
      m_useRGB(true)
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
    if (m_skinMask.size() != image.size())
    {
        m_skinMask = cv::Mat(image.size(), CV_8UC1, cv::Scalar(255, 255, 255));
    }

    if (m_dtree)
    {
        cv::Mat sample(3, 1, CV_32FC1);
        for (int y = 0; y < image.rows; ++y)
        {
            for (int x = 0; x < image.cols; ++x)
            {
                cv::Vec3b px = image.at<cv::Vec3b>(y, x);
                sample.at<float>(0, 0) = px[0];
                sample.at<float>(0, 1) = px[1];
                sample.at<float>(0, 2) = px[2];

                int response = (int)m_dtree->predict(sample);
                m_skinMask.at<uchar>(y, x) = (response == 0) ? 255 : 0;
            }
        }
    }

    cv::imshow("skinMask", m_skinMask);

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
    // Read train data: skin and non skin colors
    std::ifstream dataFile(dataPath + m_dataFileName);
    if (!dataFile.is_open())
    {
        return false;
    }

    int line[4] = { 0 };
    std::vector<float> trainColors;
    trainColors.reserve(3 * 245058);
    std::vector<int> trainMarkers;
    trainMarkers.reserve(245058);
    while (dataFile >> line[0] >> line[1] >> line[2] >> line[3])
    {
        trainColors.push_back(line[0]);
        trainColors.push_back(line[1]);
        trainColors.push_back(line[2]);
        trainMarkers.push_back(line[3] - 1);
    }
    dataFile.close();

    // Prepare data
    cv::Mat samples = cv::Mat(trainColors).reshape(1, trainMarkers.size());
    cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(samples, cv::ml::ROW_SAMPLE, cv::Mat(trainMarkers));

    cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
    dtree->setMaxDepth(8);
    dtree->setMinSampleCount(2);
    dtree->setUseSurrogates(false);
    dtree->setCVFolds(0); // the number of cross-validation folds
    dtree->setUse1SERule(false);
    dtree->setTruncatePrunedTree(false);
    dtree->train(trainData);
    m_dtree = dtree;

    return true;
}
