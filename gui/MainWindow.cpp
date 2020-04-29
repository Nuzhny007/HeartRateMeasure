//-----------------------------------------------------------------------------
#include "MainWindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QPushButton>
#include <QLabel>
#include <QStringList>
#include <QMenu>
#include <QMenuBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QEvent>
#include <QGraphicsSceneEvent>
#include <qguiapplication.h>
#include <qscreen.h>

#include <opencv2/opencv.hpp>
#include "UtilityFunctions.h"
//----------------------------------------------------------------------------- 

MainWindow::MainWindow(QString appDirPath, QWidget* parent /* = NULL */, Qt::WindowFlags f /* = 0 */)
	:
        QMainWindow(parent, f),
	    m_appDirPath(appDirPath),
        m_directionForward(true),
        m_startFrame(0),
        m_sliderByUserAction(false),
        m_inSliderMove(false),
        m_inNextFrame(false),
        m_inOpenMode(false),
	    m_mainProc(appDirPath.toStdString())
{
	setWindowTitle(QString::fromLocal8Bit("VitaGraph"));

	Qt::WindowFlags wflags = windowFlags();
	wflags |= Qt::WindowSystemMenuHint | Qt::WindowMinimizeButtonHint | Qt::WindowMaximizeButtonHint;
	wflags &= ~Qt::WindowContextHelpButtonHint | Qt::MSWindowsFixedSizeDialogHint;
	setWindowFlags(wflags);

	const QScreen* screen = QGuiApplication::primaryScreen();
	QRect gRect = QRect(0, 0, screen->size().width() / 2, screen->size().height() / 2);
	gRect.moveCenter(screen->geometry().center());
	setGeometry(gRect);
	setMinimumSize(1280 / 2, 720 / 2);

	QVBoxLayout *form_v_lay = new QVBoxLayout();
	QWidget *w = new QWidget();
	w->setLayout(form_v_lay);
	setCentralWidget(w);

    QMenu* mnu = menuBar()->addMenu(QString::fromLocal8Bit("File"));
    mnu->addAction(QString::fromLocal8Bit("Open video"), this, SLOT(OnOpenVideoFile()));
	mnu->addAction(QString::fromLocal8Bit("Open web camera"), this, SLOT(OnOpenWebCam()));
	mnu->addAction(QString::fromLocal8Bit("Open web camera settings"), this, SLOT(OnOpenWebCamSettings()));
	mnu->addSeparator();
    mnu->addAction(QString::fromLocal8Bit("Exit"), this, SLOT(close()));

    QHBoxLayout* hh_lay = new QHBoxLayout();
	form_v_lay->addLayout(hh_lay);

	QVBoxLayout* v_lay = new QVBoxLayout();
	hh_lay->addLayout(v_lay);

	QHBoxLayout* h_lay = new QHBoxLayout();
	v_lay->addLayout(h_lay);

    m_videoHeader = new QLabel();
	QFontMetrics lfm(m_videoHeader->fontMetrics());
	int labelWidth = lfm.width("1000000 x 1000000");
	int labelHeight = 2 * lfm.height();
    m_videoHeader->setText(QString::fromLocal8Bit("Video"));
    m_videoHeader->setFixedHeight(labelHeight);
    h_lay->addWidget(m_videoHeader);

    m_videoFrameSizeHeader = new QLabel();
    m_videoFrameSizeHeader->setText(QString::fromLocal8Bit("? x ?  "));
    m_videoFrameSizeHeader->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_videoFrameSizeHeader->setFixedSize(20 + labelWidth, labelHeight);
    h_lay->addWidget(m_videoFrameSizeHeader);

    m_videoFrame = new VideoFrameWidget();
    m_videoFrame->setZValue(-100000.0);
	m_plotFrame = new VideoFrameWidget();
	m_plotFrame->setZValue(-100000.0);

    m_scene = new VideoScene();
    m_scene->addItem(m_videoFrame);
	m_scene->addItem(m_plotFrame);
    m_scene->installEventFilter(this);
	
	
    m_view = new QGraphicsView();
    m_view->setScene(m_scene);
    m_view->setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
    m_view->setAlignment(Qt::AlignCenter);
    m_view->setTransformationAnchor(QGraphicsView::AnchorViewCenter);
    v_lay->addWidget(m_view);
	
	h_lay = new QHBoxLayout();
	v_lay->addLayout(h_lay);

	m_heartRate = new QLabel();
	m_heartRate->setText(QString::fromLocal8Bit("Measuring..."));
	m_heartRate->setFixedHeight(2 * m_heartRate->fontMetrics().height());
	h_lay->addWidget(m_heartRate);

	h_lay = new QHBoxLayout();
	v_lay->addLayout(h_lay);

	QPushButton* btn = new QPushButton();
	QFontMetrics bfm(btn->fontMetrics());
	int btnWidth = 2 * bfm.width("<||");
	int btnHeight = 2 * bfm.height();
	btn->setFixedSize(btnWidth, btnHeight);
	btn->setText("<||");
	h_lay->addWidget(btn);
	connect(btn, SIGNAL(clicked()), this, SLOT(OnFrameRewind()));
	btn->setEnabled(false);
	btn->setStyleSheet(QString::fromUtf8("QPushButton:disabled"
		"{ color: gray }"));

	btn = new QPushButton();
	btn->setFixedSize(btnWidth, btnHeight);
	btn->setText("<");
	h_lay->addWidget(btn);
	connect(btn, SIGNAL(clicked()), this, SLOT(OnPlayRewind()));
	btn->setDisabled(true);

	btn = new QPushButton();
	btn->setFixedSize(btnWidth, btnHeight);
	btn->setText("||");
	h_lay->addWidget(btn);
	connect(btn, SIGNAL(clicked()), this, SLOT(OnPause()));

	btn = new QPushButton();
	btn->setFixedSize(btnWidth, btnHeight);
	btn->setText(">");
	h_lay->addWidget(btn);
	connect(btn, SIGNAL(clicked()), this, SLOT(OnPlayForward()));

	btn = new QPushButton();
	btn->setFixedSize(btnWidth, btnHeight);
	btn->setText("||>");
	h_lay->addWidget(btn);
	connect(btn, SIGNAL(clicked()), this, SLOT(OnFrameForward()));
	btn->setDisabled(true);

    m_timeLineSlider = new QSlider();
    m_timeLineSlider->setMinimum(0);
    m_timeLineSlider->setMaximum(0);
    m_timeLineSlider->setSliderPosition(0);
    m_timeLineSlider->setOrientation(Qt::Horizontal);
    connect(m_timeLineSlider, SIGNAL(sliderPressed()), this, SLOT(OnSliderPressed()));
    connect(m_timeLineSlider, SIGNAL(sliderReleased()), this, SLOT(OnSliderReleased()));
    connect(m_timeLineSlider, SIGNAL(valueChanged(int)), this, SLOT(OnPositionChanged(int)));
    h_lay->addWidget(m_timeLineSlider);
	
	v_lay = new QVBoxLayout();
	hh_lay->addLayout(v_lay);

	QFormLayout* f_lay = new QFormLayout();
	v_lay->addLayout(f_lay);

    m_playTimer = new QTimer(this);
    m_playTimer->setSingleShot(true); // так как время каждый раз вычисляется заново, чтобы видео не убегало
    connect(m_playTimer, SIGNAL(timeout()), this, SLOT(OnTimer()));

    m_shortcutOpenVideoFile = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_O), this, SLOT(OnOpenVideoFile()));
	m_shortcutOpenWebCam = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_W), this, SLOT(OnOpenWebCam()));
	m_shortcutOpenWebCamSettings = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_S), this, SLOT(OnOpenWebCamSettings()));
    m_shortcutExit = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_Q), this, SLOT(close()));
}
//-----------------------------------------------------------------------------

MainWindow::~MainWindow()
{
}
//-----------------------------------------------------------------------------

void MainWindow::OnPlayForward()
{
	StartPlay(true);
}
//-----------------------------------------------------------------------------

void MainWindow::OnPlayRewind()
{
	StartPlay(false);
}
//-----------------------------------------------------------------------------

void MainWindow::OnFrameForward()
{
    m_playTimer->stop();
	NextFrame(true, false);
}
//-----------------------------------------------------------------------------

void MainWindow::OnFrameRewind()
{
    m_playTimer->stop();
	NextFrame(false, false);
}
//-----------------------------------------------------------------------------

void MainWindow::OnSliderPressed()
{
    m_sliderByUserAction = true;
}
//-----------------------------------------------------------------------------

void MainWindow::OnSliderReleased()
{
    m_sliderByUserAction = false;
}
//-----------------------------------------------------------------------------

void MainWindow::OnPositionChanged(int /*value*/)
{
	if (m_inSliderMove)
	{
		return;
	}
    m_inSliderMove = true;
	NextFrame(true, true);
    m_inSliderMove = false;
}
//-----------------------------------------------------------------------------

void MainWindow::OnOpenVideoFile()
{
    std::ifstream ifold("video_folder.txt");
    if (ifold.is_open())
    {
        std::string fstr;
        ifold >> fstr;
        ifold.close();
        m_videoFolder = fstr.c_str();
    }

    QString str = QFileDialog::getOpenFileName(this,
                                               QString::fromLocal8Bit("Open video"),
                                               m_videoFolder,
                                               "Video (*.avi *.mp4 *.mkv *.mpg *.mpv *.mov);; All files (*.*)");
	if (str.isEmpty())
		return;

    m_videoFolder = str.left(str.lastIndexOf(QRegExp("[\\/]")));
    std::ofstream ofold("video_folder.txt");
    if (ofold.is_open())
    {
        ofold << m_videoFolder.toStdString();
        ofold.close();
    }
	
	std::string confFileName((m_appDirPath + QDir::separator() + "data" + QDir::separator() + "pca_128.conf").toStdString());
	if (!m_settings.ParseOptions(confFileName))
	{
		QMessageBox::critical(this, QString::fromLocal8Bit("Config file do not opened"),
			QString("Config file %1 not supported").arg(QString(confFileName.c_str())));
		return;
	}

	if (m_settings.m_useEmotionsRecognition)
	{
		m_emotionsRecognizer = std::unique_ptr<EmotionRecognition>(CreateRecognizer(m_appDirPath.toStdString()));
	}
	else
	{
		m_emotionsRecognizer = nullptr;
	}

	m_webCam = false;

	m_settings.m_useFPS = true;
	m_settings.m_fps = 25;
	m_settings.m_freq = cv::getTickFrequency();
    if (!OpenCapture(str.toStdString(), m_capture, m_settings.m_useFPS, m_settings.m_freq, m_settings.m_fps, m_settings.m_cameraBackend))
	{
        QMessageBox::critical(this, QString::fromLocal8Bit("File do not opened"),
                              QString("File format %1 not supported").arg(str));
		return;
	}
	m_mainProc.Init(m_settings, str.toStdString());

    m_videoName = str;
    int videoLength = m_capture.get(cv::CAP_PROP_FRAME_COUNT);
	m_timeLineSlider->setEnabled(true);
	m_timeLineSlider->setMaximum(videoLength);
    m_timeLineSlider->setValue(0);
    m_timeLineSlider->setSliderPosition(0);

#if 1
    cv::Mat firstFrame;
    m_capture >> firstFrame;
    int w = firstFrame.cols;
    int h = firstFrame.rows;
    m_capture.set(cv::CAP_PROP_POS_FRAMES, 0);
#else
    int w = Capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = Capture.get(cv::CAP_PROP_FRAME_HEIGHT);
#endif

    m_scene->setSceneRect(0.0, 0.0, w, h);

    m_videoFrameSizeHeader->setText(QString("%1 x %2  ").arg(w).arg(h));

	SetSceneScale(w, h);

	StartPlay(true);
}

//-----------------------------------------------------------------------------

void MainWindow::OnOpenWebCam()
{
	std::string confFileName((m_appDirPath + QDir::separator() + "data" + QDir::separator() + "pca_128.conf").toStdString());
	if (!m_settings.ParseOptions(confFileName))
	{
		QMessageBox::critical(this, QString::fromLocal8Bit("Config file do not opened"),
			QString("Config file %1 not supported").arg(QString(confFileName.c_str())));
		return;
	}

	if (m_settings.m_useEmotionsRecognition)
	{
		m_emotionsRecognizer = std::unique_ptr<EmotionRecognition>(CreateRecognizer(m_appDirPath.toStdString()));
	}
	else
	{
		m_emotionsRecognizer = nullptr;
	}

	QString str("0");

	m_webCam = true;

	m_settings.m_useFPS = true;
	m_settings.m_fps = 25;
	m_settings.m_freq = cv::getTickFrequency();
	if (!OpenCapture(str.toStdString(), m_capture, m_settings.m_useFPS, m_settings.m_freq, m_settings.m_fps, m_settings.m_cameraBackend))
	{
		QMessageBox::critical(this, QString::fromLocal8Bit("web camera do not opened"),
			QString("Web camera %1 not found").arg(str));
		return;
	}
	m_mainProc.Init(m_settings, str.toStdString());

	m_videoName = "Web camera " + str;
	m_timeLineSlider->setEnabled(true);
	m_timeLineSlider->setMaximum(m_settings.m_fps * 60 * 10);
	m_timeLineSlider->setValue(0);
	m_timeLineSlider->setSliderPosition(0);

#if 1
	cv::Mat firstFrame;
	m_capture >> firstFrame;
	int w = firstFrame.cols;
	int h = firstFrame.rows;
#else
	int w = m_capture.get(cv::CAP_PROP_FRAME_WIDTH);
	int h = m_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
#endif

	m_scene->setSceneRect(0.0, 0.0, w, h);

	m_videoFrameSizeHeader->setText(QString("%1 x %2, %3  ").arg(w).arg(h).arg(m_settings.m_fps));

	SetSceneScale(w, h);

	StartPlay(true);
}
//-----------------------------------------------------------------------------

void MainWindow::OnOpenWebCamSettings()
{
	if (m_webCam && m_capture.isOpened())
	{
		m_capture.set(cv::CAP_PROP_SETTINGS, 1);
	}
}
//-----------------------------------------------------------------------------

void MainWindow::OnHelp()
{
}
//-----------------------------------------------------------------------------

void MainWindow::ShowFrame()
{
    m_videoFrame->SetImage(m_currentFrame);
    m_videoFrame->update();
	m_plotFrame->SetImage(m_signalPlot);
	m_plotFrame->update();
}
//-----------------------------------------------------------------------------

void MainWindow::NextFrame(bool forward, bool set_prop)
{
    if (m_inNextFrame)
		return;

    if (!m_capture.isOpened())
		return;

    m_inNextFrame = true;
	int slider = m_timeLineSlider->sliderPosition();

	if (forward)
	{
        ++slider;
        int max = m_timeLineSlider->maximum();
		if (slider < max)
		{
			if (set_prop && !m_webCam)
			{
				//m_capture.set(cv::CAP_PROP_POS_FRAMES, slider);
			}
            m_timeLineSlider->setSliderPosition(slider);
            m_capture >> m_currentFrame;
		}
        else if (m_directionForward)
		{
			m_currentFrame = cv::Mat();
            m_playTimer->stop();
		}
	}
	else
	{
        --slider;
		if (slider >= 0)
		{
			if (!m_webCam)
			{
				m_capture.set(cv::CAP_PROP_POS_FRAMES, slider);
			}
            m_timeLineSlider->setSliderPosition(slider);
            m_capture >> m_currentFrame;
		}
        else if (!m_directionForward)
		{
			m_currentFrame = cv::Mat();
            m_playTimer->stop();
		}
	}

	int frameInd = m_webCam ? slider : m_capture.get(cv::CAP_PROP_POS_FRAMES);

	if (!m_currentFrame.empty())
	{
		int64 t1 = cv::getTickCount();
		int64 captureTime = m_settings.m_useFPS ? ((frameInd * 1000.) / m_settings.m_fps) : t1;

		cv::Mat frame;
		cv::Scalar colorVal;
		bool res = m_mainProc.Process(m_currentFrame, frame, captureTime, colorVal, false, false, true, false);
		//int64 t2 = cv::getTickCount();
		if (res)
		{
			cv::Mat freqPlot(m_currentFrame.rows / 8, m_currentFrame.cols / 2, CV_8UC3, cv::Scalar::all(0));
			cv::Mat signalPlot(freqPlot.rows, freqPlot.cols, freqPlot.type());
			if (!m_mainProc.DrawSignal(signalPlot, false, false))
			{
				signalPlot = cv::Mat();
			}
			if (!m_mainProc.DrawFrequency(freqPlot))
			{
				freqPlot = cv::Mat();
			}
			DrawResult(frameInd, freqPlot, signalPlot);
		}
		else
		{
			m_heartRate->setText(QString::fromLocal8Bit("Heart rate frequency: %1").arg("face not found"));
			m_signalPlot = cv::Mat();
		}

		m_videoHeader->setText(QString::fromLocal8Bit("Video: %1 (%2 : %3)").arg(m_videoName).arg(frameInd).arg(m_timeLineSlider->maximum()));
		ShowFrame();
	}
	else
	{
		OnPause();
	}

    m_inNextFrame = false;
}
//-----------------------------------------------------------------------------

void MainWindow::DrawResult(int frameInd, const cv::Mat& freqPlot, const cv::Mat& signalPlot)
{
	std::map<EmotionRecognition::Emotions, std::string> emotions;
	emotions[EmotionRecognition::Neutral] = "Neutral";
	emotions[EmotionRecognition::Happy] = "Happy";
	emotions[EmotionRecognition::Sad] = "Sad";
	emotions[EmotionRecognition::Surprise] = "Surprise";
	emotions[EmotionRecognition::Anger] = "Anger";

	cv::Rect faceRect(m_mainProc.GetFaceRect());
	EmotionRecognition::Emotions currEmo = EmotionRecognition::Neutral;
	if (m_settings.m_useEmotionsRecognition)
	{
		currEmo = m_emotionsRecognizer->Recognition(m_currentFrame(faceRect));
	}

	int measure = m_mainProc.RemainingMeasurements();
	if (measure > 0)
	{
		m_heartRate->setText(QString::fromLocal8Bit("Heart rate frequency: waiting for the %1 frames").arg(measure));
		m_signalPlot = cv::Mat();
	}
	else
	{
		double freqMean = 0;
		double freqDev = 0;
		FrequencyResults freqResults;
		m_mainProc.GetFrequency(&freqResults, &freqMean, &freqDev);
		if (freqResults.snr == 0 || freqResults.snr > m_settings.m_snrThresold)
		{
			QString text = QString::fromLocal8Bit("Heart rate frequency: %1 (mean = %2, dev = %3").arg(cvRound(freqResults.smootFreq)).arg(freqMean).arg(freqDev);
			if (freqResults.snr > 0)
			{
				text = QString::fromLocal8Bit("%1, snr = %2").arg(text).arg(freqResults.snr);
			}
			if (freqResults.averageCardiointerval > 0)
			{
				text = QString::fromLocal8Bit("%1, HR(HRV) = %2, interval = %3 ms").arg(text).arg(60000. / freqResults.averageCardiointerval).arg(freqResults.currentCardiointerval);
			}
			text += ")";

			m_heartRate->setText(text);
		}

		if (!freqPlot.empty() && !signalPlot.empty())
		{
			m_currentFrame(cv::Rect(m_currentFrame.cols - freqPlot.cols, 0, freqPlot.cols, freqPlot.rows)) *= 0.5;
			m_currentFrame(cv::Rect(m_currentFrame.cols - freqPlot.cols, 0, freqPlot.cols, freqPlot.rows)) += 0.5 * freqPlot;
			m_currentFrame(cv::Rect(0, 0, signalPlot.cols, signalPlot.rows)) *= 0.5;
			m_currentFrame(cv::Rect(0, 0, signalPlot.cols, signalPlot.rows)) += 0.5 * signalPlot;

			std::string text = std::to_string(cvRound(freqResults.smootFreq));
			int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
			double fontScale = 0.7;
			int thickness = 2;
			int baseline = 0;
			cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
			textSize.width = std::min(120, textSize.width);
			cv::Point textPos(signalPlot.cols - textSize.width, textSize.height + 10);
			cv::putText(m_currentFrame, text, textPos, fontFace, fontScale, cv::Scalar::all(255), thickness);

			if (frameInd - m_lastCircleInd < cvRound(m_settings.m_fps) / 8)
			{
				int radius = 15;
				cv::circle(m_currentFrame, cv::Point(signalPlot.cols - radius - 2, textPos.y + 10 + radius), 15, cv::Scalar(0, 0, 255), cv::FILLED);
			}
			else if (frameInd - m_lastCircleInd > cvRound(m_settings.m_fps) / 4)
			{
				m_lastCircleInd = frameInd;
			}
		}

		if (m_settings.m_useEmotionsRecognition)
		{
			std::string text = emotions[currEmo];
			double fontScale = 1.0;
			int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
			int thickness = 2;
			int baseline = 0;
			cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
			textSize.width = std::min(120, textSize.width);
			cv::Point textPos(std::min(faceRect.x + faceRect.width, m_currentFrame.cols - textSize.width), faceRect.y + textSize.height + 10);
			cv::putText(m_currentFrame, text, textPos, fontFace, fontScale, cv::Scalar::all(255), thickness);
		}
	}
}
//-----------------------------------------------------------------------------

void MainWindow::resizeEvent(
	QResizeEvent* //event
)
{
	ShowFrame();
	SetSceneScale(m_currentFrame.cols, m_currentFrame.rows);
}
//-----------------------------------------------------------------------------

void MainWindow::showEvent(
	QShowEvent* //event
)
{
	SetSceneScale(m_currentFrame.cols, m_currentFrame.rows);
}
//-----------------------------------------------------------------------------

void MainWindow::StartPlay(bool forward)
{
	NextFrame(true, false);
    m_startFrame = m_timeLineSlider->sliderPosition();
    m_directionForward = forward;
    m_playTime.restart();
	NextTime();
}
//-----------------------------------------------------------------------------

void MainWindow::NextTime()
{
    if (!m_capture.isOpened())
		return;

    double fps = m_settings.m_fps / 1000.0;
    qint64 elapsed = m_playTime.elapsed();
 	int frames = (int)(double)(elapsed * fps);
 	int next_time =  (int)(double)((frames + 1) / fps) - elapsed;
    m_playTimer->start(next_time);
}
//-----------------------------------------------------------------------------

void MainWindow::OnPause()
{
    m_playTimer->stop();
}
//-----------------------------------------------------------------------------

void MainWindow::OnTimer()
{
    NextFrame(m_directionForward, false);
	NextTime();
}
//-----------------------------------------------------------------------------

void MainWindow::SetSceneScale(int width, int height)
{
	double scale = 1.0;
	if (width > 0 && height > 0)
	{
        QSizeF new_s = VideoFrameWidget::GetNewSize(QSizeF(width, height), QSizeF(m_view->width() - 2, m_view->height() - 2));
		scale = new_s.width() / (double)width;
	}
	QTransform tr = QTransform().scale(scale, scale);
    m_view->setTransform(tr);
}
//-----------------------------------------------------------------------------

bool MainWindow::eventFilter(QObject *obj, QEvent *event)
{
	auto set_default_cursor = [this]()
	{
        QList<QGraphicsItem*> items = m_scene->items();
		for (int i = 0; i < items.size(); i++)
		{
			QGraphicsRectItem *r_i = dynamic_cast<QGraphicsRectItem*>(items[i]);
			if (r_i)
			{
				r_i->setCursor(QCursor(Qt::ArrowCursor));
			}
		}
	};

	if (event->type() == QEvent::GraphicsSceneMousePress)
	{
		//QGraphicsSceneMouseEvent *ev = static_cast<QGraphicsSceneMouseEvent*>(event);
		return true;
	}
	else if (event->type() == QEvent::GraphicsSceneMouseRelease)
	{
		return true;
	}
	else if (event->type() == QEvent::GraphicsSceneMouseMove)
	{
		QGraphicsSceneMouseEvent *ev = static_cast<QGraphicsSceneMouseEvent*>(event);
		QPointF ev_pos = ev->scenePos();

		return true;
	}
	else if (event->type() == QEvent::GraphicsSceneMouseDoubleClick)
	{
		return true;
	}
	else
	{
		// standard event processing
		return QMainWindow::eventFilter(obj, event);
	}
}
//-----------------------------------------------------------------------------

int MainWindow::CurrentFrameIdx()
{
    return m_timeLineSlider->sliderPosition();
}
//-----------------------------------------------------------------------------

int MainWindow::MaxFrameIdx()
{
    return m_timeLineSlider->maximum();
}
//-----------------------------------------------------------------------------
