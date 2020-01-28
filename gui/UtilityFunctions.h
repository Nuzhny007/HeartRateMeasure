//-----------------------------------------------------------------------------
#ifndef __UtilityFunctions_h__
#define __UtilityFunctions_h__
//-----------------------------------------------------------------------------
#include <QImage>
#include <opencv2/opencv.hpp>
//-----------------------------------------------------------------------------

QImage  CvMatToQImage(const cv::Mat &inmat);
cv::Mat QImageToCvMat(const QImage &in_image, bool in_clone_image_data);

//-----------------------------------------------------------------------------
#endif // __UtilityFunctions_h__
//-----------------------------------------------------------------------------
