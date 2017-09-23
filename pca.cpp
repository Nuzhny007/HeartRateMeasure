#include "pca.h"

#include <numeric>

///
/// \brief MakePCA
/// \param images
/// \param resImg
/// \return
///
bool MakePCA(
	const std::deque<cv::Mat>& images,
	cv::Mat& resImg
	)
{
	if (images.size() < 2)
	{
		return false;
	}

	const int imgWidth = images[0].cols;
	const int imgHeight = images[0].rows;

	for (size_t i = 1; i < images.size(); ++i)
	{
		if (imgWidth != images[i].cols ||
			imgHeight != images[i].rows ||
			images[i].type() != CV_MAKETYPE(CV_8U, 1))
		{
			return false;
		}
	}

	const int featuresCount = images.size();
	int vectorsCount = imgHeight * imgWidth;
    cv::Mat features(vectorsCount, featuresCount, CV_32F);

    int idx = 0;
    for (int y = 0 ; y < imgHeight; ++y)
    {
		for (int x = 0; x < imgWidth; ++x)
		{
			for (int i = 0; i < static_cast<int>(images.size()); ++i)
			{
				features.at<float>(idx, i) = images[i].at<uchar>(y, x);
			}
            ++idx;
        }
    }

    const int componentsCount = 3;
    cv::PCA pca(features, cv::Mat(), CV_PCA_DATA_AS_ROW, componentsCount);

    idx = 0;
    float minV = std::numeric_limits<float>::max();
    float maxV = -std::numeric_limits<float>::max();

    for (int row = 0 ; row < imgHeight; ++row)
    {
        for (int col = 0; col < imgWidth; ++col)
        {
            cv::Mat projected = pca.project(features.row(idx));
            float val = projected.at<float>(0, 0);
            minV = std::min(minV, val);

            maxV = std::max(maxV, val);

            idx++;
        }
    }

	if (resImg.cols != imgWidth ||
		resImg.rows != imgHeight)
    {
		resImg.create(imgHeight, imgWidth, CV_8UC1);
    }
    idx = 0;
    for (int row = 0 ; row < imgHeight; ++row)
    {
        for (int col = 0; col < imgWidth; ++col)
        {
            cv::Mat projected = pca.project(features.row(idx));
            ++idx;

            float value = projected.at<float>(0, 0);
            value = 255.f * (value - minV) / (maxV - minV);
			resImg.at<uchar>(row, col) = cv::saturate_cast<uchar>(value);
        }
    }

	return true;
}

///
/// \brief MakePCA
/// \param src
/// \param dst
/// \return
///
bool MakePCA(const cv::Mat& src, cv::Mat& dst)
{
    const int vectorsCount = src.cols;

    const int componentsCount = 3;
    cv::PCA pca(src, cv::Mat(), CV_PCA_DATA_AS_COL, componentsCount);

    double minV = std::numeric_limits<double>::max();
    double maxV = -std::numeric_limits<double>::max();

    for (int idx = 0; idx < vectorsCount; ++idx)
    {
        cv::Vec3d feature(src.at<double>(0, idx), src.at<double>(1, idx), src.at<double>(2, idx));
        cv::Mat projected = pca.project(feature);
        double val = projected.at<double>(0, 0);
        minV = std::min(minV, val);

        maxV = std::max(maxV, val);
    }

    dst.create(1, vectorsCount, CV_64FC1);

    for (int idx = 0; idx < vectorsCount; ++idx)
    {
        cv::Vec3d feature(src.at<double>(0, idx), src.at<double>(1, idx), src.at<double>(2, idx));
        cv::Mat projected = pca.project(feature);

        double value = projected.at<double>(0, 0);
        value = 255.f * (value - minV) / (maxV - minV);
        dst.at<double>(0, idx) = value;
    }

    return true;
}
