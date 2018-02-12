#include "SkinDetector.h"
#include <fstream>

///
/// \brief SkinDetector::SkinDetector
///
SkinDetector::SkinDetector()
    :
      m_dataFileName("Skin_NonSkin.txt"),
      m_modelFileName("skin_model.yaml"),
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
cv::Mat SkinDetector::Detect(cv::Mat image, bool saveResult, int frameInd)
{
    if (m_skinMask.size() != image.size())
    {
        m_skinMask = cv::Mat(image.size(), CV_8UC1, cv::Scalar(255, 255, 255));
    }

    if (m_model)
    {
        cv::Mat sample(1, 3, CV_32FC1);
        for (int y = 0; y < image.rows; ++y)
        {
            const uchar* imgPtr = image.ptr(y);
            uchar* maskPtr = m_skinMask.ptr(y);

            for (int x = 0; x < image.cols; ++x)
            {
                sample.at<float>(0, 0) = imgPtr[0];
                sample.at<float>(0, 1) = imgPtr[1];
                sample.at<float>(0, 2) = imgPtr[2];

                int response = (int)m_model->predict(sample);
                *maskPtr = (response == 0) ? 255 : 0;

                imgPtr += 3;
                ++maskPtr;
            }
        }
        cv::imshow("skinMask", m_skinMask);

        if (saveResult)
        {
            std::string fileName = "skinMask/" + std::to_string(frameInd) + ".png";
            cv::imwrite(fileName, m_skinMask);
        }
    }

    return m_skinMask;
}

///
/// \brief SkinDetector::InitModel
/// \param modelPath
/// \return
///
bool SkinDetector::InitModel(std::string modelPath)
{
    try
    {
        cv::Ptr<cv::ml::DTrees> dtree = cv::Algorithm::load<cv::ml::DTrees>(modelPath + m_modelFileName);
        if (dtree)
        {
            m_model = dtree;
        }
    }
    catch(...)
    {

    }

    return m_model != 0;
}

///
/// \brief SkinDetector::SaveModel
/// \param modelPath
/// \return
///
bool SkinDetector::SaveModel(std::string modelPath)
{
    if (m_model)
    {
        m_model->save(modelPath + m_modelFileName);
        return true;
    }
    return false;
}

///
/// \brief SkinDetector::LearnModel
/// \param dataPath
/// \return
///
bool SkinDetector::LearnModel(std::string dataPath)
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
    m_model = dtree;

    return true;
}
