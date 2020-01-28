#include "EmotionDetection.h"

#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>

#include <inference_engine.hpp>

#include <opencv2/opencv.hpp>

#include "slog.hpp"
#include "common.hpp"
#include "extension/ext_list.hpp"
#include "../common/common.h"

class BaseDetection
{
public:
	InferenceEngine::ExecutableNetwork m_net;
	InferenceEngine::InferencePlugin * m_plugin;
	InferenceEngine::InferRequest::Ptr m_request;
	std::string m_topoName;
	std::string m_pathToModel;
	std::string m_deviceForInference;
	const int m_maxBatch;
	bool m_isBatchDynamic;
	const bool m_isAsync;

	BaseDetection(std::string topoName, std::string pathToModel, std::string deviceForInference, int maxBatch, bool isBatchDynamic, bool isAsync = false)
		: m_topoName(topoName), m_pathToModel(pathToModel), m_deviceForInference(deviceForInference), m_maxBatch(maxBatch), m_isAsync(isAsync)
	{
		if (m_isAsync)
		{
			slog::info << "Use async mode for " << m_topoName << slog::endl;
		}
	}

	virtual ~BaseDetection()
	{
	}

	InferenceEngine::ExecutableNetwork* operator ->()
	{
		return &m_net;
	}
	virtual InferenceEngine::CNNNetwork read() = 0;

	virtual void submitRequest()
	{
		if (!enabled() || m_request == nullptr)
			return;
		if (m_isAsync)
		{
			m_request->StartAsync();
		}
		else
		{
			m_request->Infer();
		}
	}

	virtual void wait()
	{
		if (!enabled() || !m_request || !m_isAsync)
			return;
		m_request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
	}

	bool enabled() const
	{
		if (!m_enablingChecked)
		{
			m_enabled = !m_pathToModel.empty();
			if (!m_enabled)
			{
				slog::info << m_topoName << " DISABLED" << slog::endl;
			}
			m_enablingChecked = true;
		}
		return m_enabled;
	}
	void printPerformanceCounts()
	{
		if (!enabled())
		{
			return;
		}
		slog::info << "Performance counts for " << m_topoName << slog::endl << slog::endl;
		::printPerformanceCounts(m_request->GetPerformanceCounts(), std::cout, false);
	}

protected:
	mutable bool m_enablingChecked = false;
	mutable bool m_enabled = false;
};

///
class EmotionsDetectionClass : public BaseDetection
{
public:
	std::string m_input;
	std::string m_outputEmotions;
	int m_enquedFaces = 0;

	EmotionsDetectionClass(const std::string &pathToModel, const std::string &deviceForInference)
		:
		BaseDetection("Emotions Recognition", pathToModel, deviceForInference, 16, false, false)
	{
	}

	void submitRequest()
	{
		if (!m_enquedFaces)
			return;
		if (m_isBatchDynamic)
		{
			m_request->SetBatch(m_enquedFaces);
		}
		BaseDetection::submitRequest();
		m_enquedFaces = 0;
	}

	void enqueue(const cv::Mat &face)
	{
		if (!enabled())
		{
			return;
		}
		if (m_enquedFaces == m_maxBatch)
		{
			slog::warn << "Number of detected faces more than maximum(" << m_maxBatch << ") processed by Emotions detector" << slog::endl;
			return;
		}
		if (!m_request)
		{
			m_request = m_net.CreateInferRequestPtr();
		}

		InferenceEngine::Blob::Ptr inputBlob = m_request->GetBlob(m_input);

		matU8ToBlob<uint8_t>(face, inputBlob, m_enquedFaces);

		m_enquedFaces++;
	}

	int operator[] (int idx) const
	{
		// Vector of supported emotions.
		static const std::vector<std::string> emotionsVec = { "neutral", "happy", "sad", "surprise", "anger" };
		auto emotionsVecSize = emotionsVec.size();

		InferenceEngine::Blob::Ptr emotionsBlob = m_request->GetBlob(m_outputEmotions);

		/* emotions vector must have the same size as number of channels
		 * in model output. Default output format is NCHW so we check index 1. */
		int numOfChannels = emotionsBlob->getTensorDesc().getDims().at(1);
		if (numOfChannels != emotionsVec.size())
		{
			throw std::logic_error("Output size (" + std::to_string(numOfChannels) +
				") of the Emotions Recognition network is not equal "
				"to used emotions vector size (" +
				std::to_string(emotionsVec.size()) + ")");
		}

		auto emotionsValues = emotionsBlob->buffer().as<float *>();
		auto outputIdxPos = emotionsValues + idx;

		/* we identify an index of the most probable emotion in output array
		   for idx image to return appropriate emotion name */
		int maxProbEmotionIx = std::max_element(outputIdxPos, outputIdxPos + emotionsVecSize) - outputIdxPos;
#if 0
		return emotionsVec[maxProbEmotionIx];
#else
		return maxProbEmotionIx;
#endif
	}

	InferenceEngine::CNNNetwork read()
	{
		slog::info << "Loading network files for Emotions recognition" << slog::endl;
		InferenceEngine::CNNNetReader netReader;
		// Read network model.
		netReader.ReadNetwork(m_pathToModel);

		// Set maximum batch size.
		netReader.getNetwork().setBatchSize(m_maxBatch);
		slog::info << "Batch size is set to " << netReader.getNetwork().getBatchSize() << " for Emotions recognition" << slog::endl;


		// Extract model name and load its weights.
		std::string binFileName = fileNameNoExt(m_pathToModel) + ".bin";
		netReader.ReadWeights(binFileName);

		// ----------------------------------------------------------------------------------------------

		// Emotions recognition network should have one input and one output.
		// ---------------------------Check inputs ------------------------------------------------------
		slog::info << "Checking Emotions Recognition inputs" << slog::endl;
		InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
		if (inputInfo.size() != 1)
		{
			throw std::logic_error("Emotions Recognition topology should have only one input");
		}
		auto& inputInfoFirst = inputInfo.begin()->second;
		inputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
		inputInfoFirst->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
		m_input = inputInfo.begin()->first;
		// -----------------------------------------------------------------------------------------------

		// ---------------------------Check outputs ------------------------------------------------------
		slog::info << "Checking Emotions Recognition outputs" << slog::endl;
		InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
		if (outputInfo.size() != 1)
		{
			throw std::logic_error("Emotions Recognition network should have one output layer");
		}
		for (auto& output : outputInfo)
		{
			output.second->setPrecision(InferenceEngine::Precision::FP32);
			//output.second->setLayout(InferenceEngine::Layout::NCHW);
		}

		InferenceEngine::DataPtr emotionsOutput = outputInfo.begin()->second;

		if (!emotionsOutput)
		{
			throw std::logic_error("Emotions output data pointer is invalid");
		}

		auto emotionsCreatorLayer = emotionsOutput->getCreatorLayer().lock();

		if (!emotionsCreatorLayer)
		{
			throw std::logic_error("Emotions creator layer pointer is invalid");
		}

		if (emotionsCreatorLayer->type != "SoftMax")
		{
			throw std::logic_error("In Emotions Recognition network, Emotion layer ("
				+ emotionsCreatorLayer->name +
				") should be a SoftMax, but was: " +
				emotionsCreatorLayer->type);
		}
		slog::info << "Emotions layer: " << emotionsCreatorLayer->name << slog::endl;

		m_outputEmotions = emotionsOutput->name;

		slog::info << "Loading Emotions Recognition model to the " << m_deviceForInference << " plugin" << slog::endl;
		m_enabled = true;
		return netReader.getNetwork();
	}
};

///
struct Load
{
	BaseDetection& detector;
	
	explicit Load(BaseDetection& detector)
		: detector(detector)
	{
	}

	void Into(InferenceEngine::InferencePlugin& plg, bool enable_dynamic_batch = false) const
	{
		if (detector.enabled())
		{
			std::map<std::string, std::string> config;
			std::string pluginName = plg.GetVersion()->description;
			bool isPossibleDynBatch = (pluginName.find("MKLDNN") != std::string::npos) || (pluginName.find("clDNN") != std::string::npos);
			if (enable_dynamic_batch && isPossibleDynBatch)
			{
				config[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED] = InferenceEngine::PluginConfigParams::YES;
			}
			detector.m_net = plg.LoadNetwork(detector.read(), config);
			detector.m_plugin = &plg;
		}
	}
};

///
class EmotionRecognitionImpl : public EmotionRecognition
{
public:
	///
	EmotionRecognitionImpl()
	{
	}
	///
	~EmotionRecognitionImpl()
	{
	}

	///
	bool Init(const std::string& dataPath)
	{
		// --------------------------- 1. Load Plugin for inference engine -------------------------------------
		/// target device for face detection
		std::string d = "CPU";
		/// Define parameter for face detection model file
		std::string m = "";
		
		/// target device for AgeGender net
		std::string d_ag = "CPU";
		/// Define parameter for face detection  model file
		std::string m_ag = "";

		/// target device for HeadPose net
		std::string d_hp = "CPU";
		/// Define parameter for face detection  model file
		std::string m_hp = "";

		/// target device for Emotions net
		std::string d_em = "CPU";
		/// Define parameter for face detection model file
		std::string m_em = dataPath + "emotions-recognition-retail-0003.xml";
		slog::info << m_em << slog::endl;

		/// Absolute path to CPU library with user layers
		std::string l = "";
		/// clDNN custom kernels path
		std::string c = "";

		/// enable per-layer performance report
		bool pc = false;

		std::vector<std::pair<std::string, std::string>> cmdOptions =
		{
			{ d, m }, { d_ag, m_ag }, { d_hp, m_hp }, { d_em, m_em }
		};

		for (auto && option : cmdOptions)
		{
			auto deviceName = option.first;
			auto networkName = option.second;

			if (deviceName == "" || networkName == "")
			{
				continue;
			}

			if (m_pluginsForDevices.find(deviceName) != m_pluginsForDevices.end())
			{
				continue;
			}
			slog::info << "Loading plugin " << deviceName << slog::endl;
			InferenceEngine::InferencePlugin plugin = InferenceEngine::PluginDispatcher({ ".", "" }).getPluginByDevice(deviceName);

			/** Printing plugin version **/
			printPluginVersion(plugin, std::cout);

			/** Load extensions for the CPU plugin **/
			if ((deviceName.find("CPU") != std::string::npos))
			{
				plugin.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>());

				if (!l.empty())
				{
					// CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
					auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(l);
					plugin.AddExtension(extension_ptr);
				}
			}
			else if (!c.empty())
			{
				// Load Extensions for other plugins not CPU
				plugin.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, c} });
			}
			m_pluginsForDevices[deviceName] = plugin;
		}

		/** Per layer metrics **/
		if (pc)
		{
			for (auto && plugin : m_pluginsForDevices)
			{
				plugin.second.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES} });
			}
		}
		m_emoDetection = std::make_unique<EmotionsDetectionClass>(m_em, d_em);
		Load(*m_emoDetection).Into(m_pluginsForDevices[d_em], m_emoDetection->m_isBatchDynamic);

		return true;
	}

	///
	Emotions Recognition(const cv::Mat& face)
	{
		Emotions emo = Neutral;

		if (m_emoDetection->enabled())
		{
			m_emoDetection->enqueue(face);
			m_emoDetection->submitRequest();
			m_emoDetection->wait();

			for (int i = 0; i < m_emoDetection->m_maxBatch; ++i)
			{
				auto emotion = (*m_emoDetection)[i];
				std::cout << "Emotion: " << emotion << std::endl;
				emo = Ind2Emo(emotion);
			}
		}
		return emo;
	}

private:
	std::unique_ptr<EmotionsDetectionClass> m_emoDetection;
	std::map<std::string, InferenceEngine::InferencePlugin> m_pluginsForDevices;
};

///
EmotionRecognition* CreateRecognizer(const std::string& appDirPath)
{
	EmotionRecognitionImpl* recog = new EmotionRecognitionImpl();
	std::string dataPath = appDirPath + PathSeparator() + "data" + PathSeparator();
	if (recog->Init(dataPath))
	{
		return recog;
	}
	else
	{
		delete recog;
		return nullptr;
	}
}
