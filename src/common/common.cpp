#include "common.h"
#include <boost/program_options.hpp>
namespace po = boost::program_options;


///
/// \brief OpenCapture
/// \param fileName
/// \param capture
/// \param useFPS
/// \param freq
/// \return
///
bool OpenCapture(const std::string& fileName,
	cv::VideoCapture& capture,
	bool& useFPS,
	double& freq,
	double& fps,
	cv::VideoCaptureAPIs cameraBackend)
{
	std::cout << "Open capture " << fileName << "..." << std::endl;
	if (fileName.size() > 1)
	{
		capture.open(fileName);
	}
	else
	{
		int cameraInd = atoi(fileName.c_str());
		std::cout << "Open camera " << cameraInd << "..." << std::endl;
		capture.open(cameraInd, cameraBackend);
		useFPS = false;
	}

	if (capture.isOpened())
	{
		int camBackedn = static_cast<int>(capture.get(cv::CAP_PROP_BACKEND));
		std::cout << "Capture was opened with backend " << camBackedn << "!" << std::endl;

		if (useFPS)
		{
			fps = capture.get(cv::CAP_PROP_FPS);
			freq = 1000.;
		}
		return true;
	}
	return false;
}

///
MeasureSettings::MeasureSettings()
{
	m_filterStr2T["pca"] = FilterPCA;
	m_filterStr2T["ica"] = FilterICA;
	m_filterStr2T["green"] = FilterGreen;

	m_filterT2Str[FilterPCA] = "pca";
	m_filterT2Str[FilterICA] = "ica";
	m_filterT2Str[FilterGreen] = "green";

	m_faceDetectorStr2T["haar"] = Haar;
	m_faceDetectorStr2T["resnet"] = Resnet;
	m_faceDetectorStr2T["vino"] = VINO;

	m_faceDetectorT2Str[Haar] = "haar";
	m_faceDetectorT2Str[Resnet] = "resnet";
	m_faceDetectorT2Str[VINO] = "vino";

	m_backendStr2T["0"] = cv::CAP_ANY;
	m_backendStr2T["vfw"] = cv::CAP_VFW;
	m_backendStr2T["dshow"] = cv::CAP_DSHOW;
	m_backendStr2T["msmf"] = cv::CAP_MSMF;
	m_backendStr2T["ffmpeg"] = cv::CAP_FFMPEG;
	
	m_backendT2Str[cv::CAP_ANY] = "0";
	m_backendT2Str[cv::CAP_VFW] = "vfw";
	m_backendT2Str[cv::CAP_DSHOW] = "dshow";
	m_backendT2Str[cv::CAP_MSMF] = "msmf";
	m_backendT2Str[cv::CAP_FFMPEG] = "ffmpeg";
}

///
bool MeasureSettings::ParseOptions(const std::string& confFileName)
{
	std::ifstream configFile(confFileName);
	if (!configFile.is_open())
	{
		return false;
	}

	po::options_description desc;
	po::variables_map variables = po::variables_map();

	desc.add_options()
		("config.camera_backend", po::value<std::string>()->default_value(m_backendT2Str[m_cameraBackend]), "Backend for web-camera")
		("config.sample_size", po::value<int>()->default_value(m_sampleSize), "Sample size (power of 2)")
		("config.motion_ampfl", po::value<int>()->default_value(m_useMA ? 1 : 0), "Use or not motion ampflification")
		("config.skin_detect", po::value<int>()->default_value(m_useSkinDetection ? 1 : 0), "Use or not skin detection")
		("config.calc_mean", po::value<int>()->default_value(m_calcMean ? 1 : 0), "Calculate mean or median color value")
		("config.emotions_recognition", po::value<int>()->default_value(m_useEmotionsRecognition ? 1 : 0), "Use emotions recognition")
		("config.filter_type", po::value<std::string>()->default_value(m_filterT2Str[m_filterType]), "Filter type: pca, ica or green")
		("config.signal_norm", po::value<int>()->default_value(m_signalNormalization ? 1 : 0), "Use normalization signal before FFT")
		("config.face_detector", po::value<std::string>()->default_value(m_faceDetectorT2Str[m_faceDetectorType]), "Face detector type: haar, resnet, vino")
		("config.gpu", po::value<int>()->default_value(m_useOCL ? 1 : 0), "Use OpenCL acceleration")
		("config.save_results", po::value<int>()->default_value(0), "Write results to disk")
		("config.use_external_control", po::value<int>()->default_value(0), "Recognize EKG values")
		("config.ma_algorithm", po::value<int>()->default_value(m_maAlgorithm), "Motion amplification algorithm: classic eulerian or simple")
		("config.ma_use_crop", po::value<int>()->default_value(m_maUseCrop ? 1 : 0), "Motion amplification: Apply only for face area")
		("config.ma_alpha", po::value<int>()->default_value(m_maAlpha), "Motion amplification parameter")
		("config.ma_lambda_c", po::value<int>()->default_value(m_maLambdaC), "Motion amplification parameter")
		("config.ma_flow", po::value<float>()->default_value(m_maFlow), "Motion amplification parameter")
		("config.ma_fhight", po::value<float>()->default_value(m_maFhight), "Motion amplification parameter")
		("config.ma_chromAttenuation", po::value<float>()->default_value(m_maChromAttenuation), "Motion amplification parameter")
		("config.return_exp_frequency", po::value<int>()->default_value(m_return_exp_frequency), "No Gaussian processiing, return raw frequency with exponential smoothing")
		("config.gauss_def_var", po::value<float>()->default_value(m_gauss_def_var), "Default variance")
		("config.gauss_min_var", po::value<float>()->default_value(m_gauss_min_var), "Minimum variance")
		("config.gauss_max_var", po::value<float>()->default_value(m_gauss_max_var), "Maximum variance")
		("config.gauss_eps", po::value<float>()->default_value(m_gauss_eps), "Model accuracy")
		("config.gauss_update_alpha", po::value<float>()->default_value(m_gauss_update_alpha), "Coefficient for mean and variance updating")
		("config.gauss_proc_alpha", po::value<float>()->default_value(m_gauss_proc_alpha), "Coefficient for updating gaussian process weight")
		("config.gauss_proc_weight_thresh", po::value<float>()->default_value(m_gauss_proc_weight_thresh), "If the weight of the Porocess is bigger then threshold then this Process is robust")
		("config.signal_lib", po::value<std::string>()->default_value(m_signalLib), "Dynamic loaded library for signal processing and Heart rate estimation")
		("config.snr_threshold", po::value<float>()->default_value(m_snrThresold), "SNR threshold for define correct measuring");

	try
	{
		po::parsed_options parsed = po::parse_config_file(configFile, desc, true);
		po::store(parsed, variables);
		po::notify(variables);

		m_cameraBackend = m_backendStr2T[variables["config.camera_backend"].as<std::string>()];

		m_useOCL = variables["config.gpu"].as<int>() != 0;

		// Use motion ampflifacation
		m_useMA = variables["config.motion_ampfl"].as<int>() != 0;

		// Measurents count
		m_sampleSize = variables["config.sample_size"].as<int>();

		m_useSkinDetection = variables["config.skin_detect"].as<int>() != 0;

		m_calcMean = variables["config.calc_mean"].as<int>() != 0;

		m_useEmotionsRecognition = variables["config.emotions_recognition"].as<int>() != 0;

		m_filterName = variables["config.filter_type"].as<std::string>();
		m_filterType = m_filterStr2T[m_filterName];
		m_signalNormalization = variables["config.signal_norm"].as<int>() != 0;

		m_faceDetectorType = m_faceDetectorStr2T[variables["config.face_detector"].as<std::string>()];
		
		m_return_exp_frequency = variables["config.return_exp_frequency"].as<int>() != 0;
		m_gauss_def_var = variables["config.gauss_def_var"].as<float>();
		m_gauss_min_var = variables["config.gauss_min_var"].as<float>();
		m_gauss_max_var = variables["config.gauss_max_var"].as<float>();
		m_gauss_eps = variables["config.gauss_eps"].as<float>();
		m_gauss_update_alpha = variables["config.gauss_update_alpha"].as<float>();
		m_gauss_proc_alpha = variables["config.gauss_proc_alpha"].as<float>();
		m_gauss_proc_weight_thresh = variables["config.gauss_proc_weight_thresh"].as<float>();

		std::map<int, MAAlgorithms> maDict;
		maDict[1] = Eulerian;
		maDict[2] = Simple;
		m_maAlgorithm = maDict[variables["config.ma_algorithm"].as<int>()];
		m_maUseCrop = maDict[variables["config.ma_use_crop"].as<int>()] != 0;
		m_maAlpha = variables["config.ma_alpha"].as<int>();
		m_maLambdaC = variables["config.ma_lambda_c"].as<int>();
		m_maFlow = variables["config.ma_flow"].as<float>();
		m_maFhight = variables["config.ma_fhight"].as<float>();
		m_maChromAttenuation = variables["config.ma_chromAttenuation"].as<float>();

		m_saveResults = variables["config.save_results"].as<int>() != 0;

		m_signalLib = variables["config.signal_lib"].as<std::string>();
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
		m_signalLib += ".dll";
#else
		m_signalLib = "lib" + m_signalLib + ".so";
#endif

		m_snrThresold = variables["config.snr_threshold"].as<float>();
	}
	catch (std::exception& ex)
	{
		std::cout << "Config file read error: " << ex.what() << std::endl;
	}
	return true;
}
