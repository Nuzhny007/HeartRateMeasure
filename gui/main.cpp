//-----------------------------------------------------------------------------
#include <QApplication>
#include "MainWindow.h"
#include <QProcessEnvironment>

#define IE_DBG 0

#if IE_DBG
#ifdef _WIN32
#include <windows.h>
#endif
#endif
//-----------------------------------------------------------------------------

int main(int argc, char** argv)
{
#if IE_DBG
#ifdef _WIN32
	if (AttachConsole(ATTACH_PARENT_PROCESS))
	{
		freopen("CONOUT$", "w", stdout);
		freopen("CONOUT$", "w", stderr);
	}
#endif
#endif

	QApplication app(argc, argv);
	app.setApplicationName(QString::fromLocal8Bit("VitaGraph"));
	app.setWindowIcon(QIcon(":/main_icon"));

	MainWindow w(app.applicationDirPath());
	w.show();
	w.update();
	
	return app.exec();
}
//-----------------------------------------------------------------------------
