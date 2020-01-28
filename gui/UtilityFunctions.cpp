//-----------------------------------------------------------------------------
#include "UtilityFunctions.h"
#include <QDebug>
//-----------------------------------------------------------------------------

QImage CvMatToQImage(const cv::Mat &inmat)
{
	switch (inmat.type())
	{
	case CV_8UC4: // 8-bit, 4 channel
		{
			QImage image(inmat.data, inmat.cols, inmat.rows, static_cast<int>(inmat.step), QImage::Format_RGB32);
			return image;
		}
	case CV_8UC3: // 8-bit, 3 channel
		{
			QImage image(inmat.data, inmat.cols, inmat.rows, static_cast<int>(inmat.step), QImage::Format_RGB888);
			return image.rgbSwapped();
		}

	case CV_8UC1: // 8-bit, 1 channel
		{
			static QVector<QRgb>  s_color_table;

			// only create our color table once
			if (s_color_table.isEmpty())
			{
				for (int i = 0; i < 256; ++i)
				{
					s_color_table.push_back(qRgb(i, i, i));
				}
			}

			QImage image(inmat.data, inmat.cols, inmat.rows, static_cast<int>(inmat.step), QImage::Format_Indexed8);
			image.setColorTable(s_color_table);
			return image;
		}
	default:
		qWarning() << "ASM::cvMatToQImage() - cv::Mat image type not handled in switch:" << inmat.type();
		break;
	}

	return QImage();
}
//-----------------------------------------------------------------------------

cv::Mat QImageToCvMat(const QImage &in_image, bool in_clone_image_data)
{
	switch (in_image.format())
	{
	case QImage::Format_RGB32: // 8-bit, 4 channel
		{
			cv::Mat mat(in_image.height(), in_image.width(), CV_8UC4, const_cast<uchar*>(in_image.bits()), in_image.bytesPerLine());
			return (in_clone_image_data ? mat.clone() : mat);
		}
	case QImage::Format_RGB888: // 8-bit, 3 channel
		{
			if (!in_clone_image_data)
				qWarning() << "ASM::QImageToCvMat() - Conversion requires cloning since we use a temporary QImage";

			QImage swapped = in_image.rgbSwapped();
			return cv::Mat(swapped.height(), swapped.width(), CV_8UC3, const_cast<uchar*>(swapped.bits()), swapped.bytesPerLine()).clone();
		}
	case QImage::Format_Indexed8: // 8-bit, 1 channel
		{
			cv::Mat mat(in_image.height(), in_image.width(), CV_8UC1, const_cast<uchar*>(in_image.bits()), in_image.bytesPerLine());
			return (in_clone_image_data ? mat.clone() : mat);
		}
	default:
		qWarning() << "ASM::QImageToCvMat() - QImage format not handled in switch:" << in_image.format();
		break;
	}

	return cv::Mat();
}
//-----------------------------------------------------------------------------
