//-----------------------------------------------------------------------------
#include "VideoScene.h"
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsRectItem>
//-----------------------------------------------------------------------------

VideoScene::VideoScene(QObject *parent) : QGraphicsScene(parent)
{
}
//-----------------------------------------------------------------------------

VideoScene::VideoScene(const QRectF &sceneRect, QObject *parent) : QGraphicsScene(sceneRect, parent)
{
}
//-----------------------------------------------------------------------------

VideoScene::VideoScene(qreal x, qreal y, qreal width, qreal height, QObject *parent) : QGraphicsScene(x, y, width, height, parent)
{
}
//-----------------------------------------------------------------------------

VideoScene::~VideoScene()
{
}
//-----------------------------------------------------------------------------
