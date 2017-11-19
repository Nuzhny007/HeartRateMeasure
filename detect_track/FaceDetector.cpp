#include "FaceDetector.h"

//------------------------------------------------------
//
//------------------------------------------------------
FaceDetector::FaceDetector()
    :
      m_kw(0.3),
      m_kh(0.3)
{
	std::string fileName = "../HeartRateMeasure/data/haarcascades/haarcascade_frontalface_alt2.xml";
    if (m_cascade.empty())
    {
		m_cascade.load(fileName);
		if (m_cascade.empty())
		{
			m_cascade.load("../" + fileName);
		}
    }
    assert(!m_cascade.empty());
}

//------------------------------------------------------
//
//------------------------------------------------------
FaceDetector::~FaceDetector()
{
}

//------------------------------------------------------
//
//------------------------------------------------------
cv::Rect FaceDetector::detect_biggest_face(cv::Mat& image, bool originalFace)
{
    cv::Rect res(0, 0, 0, 0);

	if (m_cascade.empty())
	{
		assert(0);
		return res;
	}

    bool findLargestObject = true;
    bool filterRects = true;

    cv::Mat im;
    if (image.channels() == 3)
    {
        cv::cvtColor(image, im, cv::COLOR_BGR2GRAY);
    }
    else
    {
        im = image;
    }

    std::vector<cv::Rect> faceRects;
    m_cascade.detectMultiScale(im,
                               faceRects,
                               1.2,
                               (filterRects || findLargestObject) ? 4 : 0,
                               findLargestObject ? cv::CASCADE_FIND_BIGGEST_OBJECT : 0,
                               cv::Size(image.cols / 4, image.rows / 4));
    if (!faceRects.empty())
    {
        res = faceRects[0];
    }

    if (!originalFace)
    {
        const double dw = res.width * m_kw;
        const double dh = res.height * m_kh;
        const double dx = dw / 2.0;
        const double dy = dh / 2.0;

        res.x += static_cast<int>(dx);
        res.y += static_cast<int>(dy);
        res.width -= static_cast<int>(dw);
        res.height -= static_cast<int>(dh);
    }

    return res;
}

