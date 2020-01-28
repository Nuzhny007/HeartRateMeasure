#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>

///
inline char* PathSeparator()
{
#if defined(WIN32) || defined(_WIN32) 
	static char* sep = "\\";
#else
	static char* sep = "/";
#endif

	return sep;
}

///
/// \brief OpenCapture
/// \param fileName
/// \param capture
/// \param useFPS
/// \param freq
/// \return
///
bool OpenCapture(const std::string& fileName, cv::VideoCapture& capture, bool& useFPS, double& freq, double& fps, cv::VideoCaptureAPIs cameraBackend);

///
/// \brief The MeasureSettings struct
///
struct MeasureSettings
{
	MeasureSettings();

	enum RGBFilters
	{
		FilterICA,
		FilterPCA,
		FilterGreen
	};

	enum MAAlgorithms
	{
		Unknown,
		Eulerian,
		Simple
	};

	enum FaceDetectors
	{
		Haar,
		Resnet,
		VINO
	};

	cv::VideoCaptureAPIs m_cameraBackend = cv::CAP_ANY;
	bool m_useOCL = false;
	bool m_useMA = true;
	int m_sampleSize = 128;
	bool m_useSkinDetection = true;
	bool m_calcMean = true;
	bool m_useEmotionsRecognition = false;
	double m_freq = 1000.;
	double m_fps = 0.;
	bool m_useFPS = true;
	RGBFilters m_filterType = FilterPCA;
	std::string m_filterName = "pca";
	bool m_signalNormalization = true;
	FaceDetectors m_faceDetectorType = VINO;
	bool m_return_exp_frequency = false;
	float m_gauss_def_var = 5.0f;
	float m_gauss_min_var = 2.0f;
	float m_gauss_max_var = 10.0f;
	float m_gauss_eps = 2.7f;
	float m_gauss_update_alpha = 0.1f;
	float m_gauss_proc_alpha = 0.05f;
	float m_gauss_proc_weight_thresh = 0.2f;
	MAAlgorithms m_maAlgorithm = Eulerian;
	bool m_maUseCrop = true;
	int m_maAlpha = 10;
	int m_maLambdaC = 16;
	float m_maFlow = 0.4f;
	float m_maFhight = 3.0f;
	float m_maChromAttenuation = 1.0f;
	bool m_saveResults = false;
	std::string m_signalLib = "signal0";
	float m_snrThresold = 2.5f;

	bool ParseOptions(const std::string& confFileName);

	std::map<std::string, RGBFilters> m_filterStr2T;
	std::map<RGBFilters, std::string> m_filterT2Str;

	std::map<std::string, FaceDetectors> m_faceDetectorStr2T;
	std::map<FaceDetectors, std::string> m_faceDetectorT2Str;

	std::map<std::string, cv::VideoCaptureAPIs> m_backendStr2T;
	std::map<cv::VideoCaptureAPIs, std::string> m_backendT2Str;
};

///
/// \brief The FaceCrop class
///
class FaceCrop
{
public:
	cv::Rect NewFace(const cv::Rect& faceRect, cv::Size frameSize)
	{
		if (m_cropRect.empty() || faceRect.empty())
		{
			m_cropRect = RawCrop(faceRect, frameSize);
		}
		else
		{
			if (m_cropRect.x < faceRect.x &&
				m_cropRect.x + m_cropRect.width > faceRect.x + faceRect.width &&
				m_cropRect.y < faceRect.y &&
				m_cropRect.y + m_cropRect.height > faceRect.y + faceRect.height)
			{
				// Do nothing
			}
			else
			{
				m_cropRect = RawCrop(faceRect, frameSize);
			}

		}
		m_faceRect = faceRect;
		return m_cropRect;
	}

private:
	cv::Rect m_faceRect;
	cv::Rect m_cropRect;
	static const int FacePart = 2;

	///
	cv::Rect RawCrop(const cv::Rect& faceRect, cv::Size frameSize)
	{
		cv::Rect newCrop;

		newCrop.x = faceRect.x - faceRect.width / FacePart;
		newCrop.width = faceRect.width + 2 * (faceRect.width / FacePart);

		newCrop.y = faceRect.y - faceRect.height / FacePart;
		newCrop.height = faceRect.height + 2 * (faceRect.height / FacePart);

		Clamp(newCrop.x, newCrop.width, frameSize.width);
		Clamp(newCrop.y, newCrop.height, frameSize.height);

		return newCrop;
	}

	///
	bool Clamp(int& v, int& size, int hi)
	{
		if (v < 0)
		{
			v = 0;
			if (size > hi - 1)
			{
				size = hi - 1;
			}
			return true;
		}
		else if (v + size > hi - 1)
		{
			v = hi - 1 - size;
			if (v < 0)
			{
				size += v;
				v = 0;
			}
			if (v + size > hi - 1)
			{
				size = hi - 1;
			}
			return true;
		}
		return false;
	}
};

///
template<typename T>
class StatisticLogger
{
public:
	///
	StatisticLogger()
	{
		m_history.reserve(MAX_HISTORY);
		
	}
	///
	~StatisticLogger()
	{
		FlushToFile();
	}

	///
	void Init(const std::string& fileName)
	{
		FlushToFile();

		m_fileName = fileName;
		if (!m_fileName.empty())
		{
			remove(m_fileName.c_str());
		}
		m_history.clear();
		m_sum = 0;
		m_sumSqr = 0;
		m_count = 0;
	}

	///
	void NewMeasure(size_t ind, T val)
	{
		m_history.emplace_back(ind, val);
		if (m_history.size() > MAX_HISTORY)
		{
			FlushToFile();
		}
		m_sum += val;
		m_sumSqr += val * val;
		++m_count;
	}

	///
	void GetMeanStdDev(double& mean, double& dev)
	{
		if (m_count > 1)
		{
			dev = sqrt((m_sumSqr - (m_sum * m_sum) / m_sum) / (m_sum - 1));
			mean = m_sum / m_count;
		}
		else
		{
			mean = m_sum;
			dev = 0;
		}
	}

	///
	bool FlushToFile()
	{
		bool res = false;
		if (!m_fileName.empty() && !m_history.empty())
		{
			std::fstream file;
			file.open(m_fileName, std::ios_base::in | std::ios_base::out | std::ios_base::app);
			if (file.is_open())
			{
				for (const auto& val : m_history)
				{
					file << val.first << ";" << cvRound(val.second) << "\n";
				}
				res = true;
			}
		}
		m_history.clear();
		return res;
	}

private:

	std::string m_fileName;

	double m_sum = 0;
	double m_sumSqr = 0;
	size_t m_count = 0;
	std::vector<std::pair<size_t, T>> m_history;
	static const size_t MAX_HISTORY = 200;
};
