//-----------------------------------------------------------------------------
#ifndef __VideoScene_h__
#define __VideoScene_h__
//-----------------------------------------------------------------------------
#include <QGraphicsScene>

//-----------------------------------------------------------------------------
class VideoScene : public QGraphicsScene
{
	Q_OBJECT
public:
	VideoScene(QObject *parent = 0);
	VideoScene(const QRectF &sceneRect, QObject *parent = 0);
	VideoScene(qreal x, qreal y, qreal width, qreal height, QObject *parent = 0);
	~VideoScene();
};
//-----------------------------------------------------------------------------
#endif // __VideoScene_h__
//-----------------------------------------------------------------------------
