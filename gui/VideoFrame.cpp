//-----------------------------------------------------------------------------
#include "VideoFrame.h"
#include <QPainter>
#include <QElapsedTimer>
#include "UtilityFunctions.h"
#include <opencv2/opencv.hpp>
#include <QGraphicsView>
//----------------------------------------------------------------------------- 

VideoFrameWidget::VideoFrameWidget(QWidget* /*parent*/)
{
}
//-----------------------------------------------------------------------------

VideoFrameWidget::~VideoFrameWidget()
{
}
//-----------------------------------------------------------------------------

void VideoFrameWidget::SetImage(const cv::Mat& image)
{
    m_image = image;
    QImage img = CvMatToQImage(m_image);
	QPixmap px = QPixmap::fromImage(img);
	setPixmap(px);
}
//-----------------------------------------------------------------------------

QSizeF VideoFrameWidget::GetNewSize(QSizeF old, QSizeF new_s)
{
	if (!old.height())
		return new_s;
	double aspect_ratio = (double)old.width() / (double)old.height();
	double w_new = new_s.height() * aspect_ratio;
	double h_new = new_s.width() / aspect_ratio;
	double w = 0;
	double h = 0;
	if (w_new <= new_s.width())
	{
		w = w_new;
		h = new_s.height();
	}
	else
	{
		w = new_s.width();
		h = h_new;
	}
	return QSizeF(w, h);
}
//-----------------------------------------------------------------------------
