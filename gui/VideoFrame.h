//-----------------------------------------------------------------------------
#ifndef __VideoFrame_h__
#define __VideoFrame_h__
//-----------------------------------------------------------------------------
#include <QGraphicsPixmapItem>
#include <QImage>
#include <opencv2/opencv.hpp>
//-----------------------------------------------------------------------------

class VideoFrameWidget : public QGraphicsPixmapItem
{
//	Q_OBJECT
public:
	VideoFrameWidget(QWidget* parent = NULL);
	~VideoFrameWidget();

	void SetImage(const cv::Mat& image);

	static QSizeF GetNewSize(QSizeF old, QSizeF new_s);

protected:

private:
    cv::Mat m_image;
    QSize m_size;
};

//-----------------------------------------------------------------------------
#endif // __VideoFrame_h__
//-----------------------------------------------------------------------------
