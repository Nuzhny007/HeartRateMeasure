//-----------------------------------------------------------------------------
#ifndef __MainWindow_h__
#define __MainWindow_h__
//-----------------------------------------------------------------------------
#include <QMainWindow>
#include <QLabel>
#include <QListWidget>
#include <QSlider>
#include <QComboBox>
#include <QLineEdit>
#include <opencv2/opencv.hpp>
#include <QImage>
#include <QTimer>
#include <QElapsedTimer>
#include "VideoFrame.h"
#include "VideoScene.h"
#include <QGraphicsView>
#include <QShortcut>

#include "../src/beat_calc/MainProcess.h"
#include "../src/common/common.h"
#include "../src/detect_track/EmotionDetection.h"
//-----------------------------------------------------------------------------

class MainWindow : public QMainWindow
{
	Q_OBJECT
public:
	MainWindow(QString appDirPath, QWidget* parent = NULL, Qt::WindowFlags f = 0);
	~MainWindow();

	int CurrentFrameIdx();
	int MaxFrameIdx();

public slots:
	void OnPlayForward();
	void OnPlayRewind();
	void OnFrameForward();
	void OnFrameRewind();
	void OnSliderPressed();
	void OnSliderReleased();
	void OnPositionChanged(int value);
	void OnOpenVideoFile();
	void OnOpenWebCam();
	void OnOpenWebCamSettings();
	void OnHelp();
	void OnTimer();
	void OnPause();

protected:
	void resizeEvent(QResizeEvent *event);
	void showEvent(QShowEvent *event);
	bool eventFilter(QObject *obj, QEvent *event);

private:
	QLabel* m_heartRate;
    QLabel* m_videoHeader;
    QLabel* m_videoFrameSizeHeader;
    VideoFrameWidget* m_videoFrame;
    QGraphicsView* m_view;
    VideoScene* m_scene;
	VideoFrameWidget* m_plotFrame;
	int m_lastCircleInd = 0;

    QShortcut* m_shortcutOpenVideoFile;
	QShortcut* m_shortcutOpenWebCam;
	QShortcut* m_shortcutOpenWebCamSettings;
    QShortcut* m_shortcutExit;

    QString m_videoFolder;
	QString m_appDirPath;

    QSlider* m_timeLineSlider;
    bool m_directionForward;
    QTimer* m_playTimer;
    cv::VideoCapture m_capture;
	bool m_webCam = false;
    cv::Mat m_currentFrame;
	cv::Mat m_signalPlot;

    QElapsedTimer m_playTime;
    int m_startFrame;
    bool m_sliderByUserAction;
    bool m_inSliderMove;
    bool m_inNextFrame;

    QString m_videoName;

    bool m_inOpenMode;

	MeasureSettings m_settings;
	MainProcess m_mainProc;

	std::unique_ptr<EmotionRecognition> m_emotionsRecognizer;

	void StartPlay(bool forward);
	void ShowFrame();
	void NextFrame(bool forward, bool set_prop);
	void DrawResult(int frameInd, const cv::Mat& freqPlot, const cv::Mat& signalPlot);
	void NextTime();
	void SetSceneScale(int width, int height);
};
//-----------------------------------------------------------------------------
#endif // __MainWindow_h__
//-----------------------------------------------------------------------------
