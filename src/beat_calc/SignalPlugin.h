#pragma once

#include <boost/dll/import.hpp>
#include <boost/dll/shared_library.hpp>
#include <boost/function.hpp>

#include "plugin.h"

///
class SignalPlugin
{
public:
	///
	SignalPlugin()
	{

	}
	///
	~SignalPlugin()
	{
		UnloadPlugin();
	}
	///
	bool IsLoaded() const
	{
		return m_handle != 0;
	}
	///
	bool LoadPlugin(const std::string& dllName)
	{
		m_dllName = dllName;

		try
		{
			//m_lib.load(dllName, boost::dll::load_mode::default_mode);
			m_lib.load(dllName, boost::dll::load_mode::search_system_folders);
		}
		catch (std::exception& ex)
		{
			std::cout << "Library " << dllName << " was not loaded!" << std::endl;
			std::cerr << ex.what() << std::endl;
		}

		if (m_lib.is_loaded())
		{
			try
			{
				m_CreatePlugin = m_lib.get<CreatePlugin_t>("CreatePlugin");
				m_DestroyPlugin = m_lib.get<DestroyPlugin_t>("DestroyPlugin");
				m_Reset = m_lib.get<Reset_t>("Reset");
				m_AddMeasure = m_lib.get<AddMeasure_t>("AddMeasure");
				m_MeasureFrequency = m_lib.get<MeasureFrequency_t>("MeasureFrequency");
				m_GetFrequency = m_lib.get<GetFrequency_t>("GetFrequency");
				m_RemainingMeasurements = m_lib.get<RemainingMeasurements_t>("RemainingMeasurements");
				m_GetSignal = m_lib.get<GetSignal_t>("GetSignal");
			}
			catch (std::exception& ex)
			{
				std::cout << "Some functions was not exported from " << dllName << std::endl;
				std::cerr << ex.what() << std::endl;
			}
		}
		return m_lib.is_loaded();
	}
	///
	void UnloadPlugin()
	{
		if (IsLoaded())
		{
			m_DestroyPlugin(m_handle);
			m_handle = 0;
		}
		if (m_lib.is_loaded())
		{
			m_lib.unload();
		}
	}
	///
	bool Init(const InputParams* inputParams)
	{
		m_handle = m_CreatePlugin(inputParams);

		//std::cout << "m_CreatePlugin: m_handle = " << m_handle << std::endl;

		return IsLoaded();
	}

	///
	void Reset()
	{
		m_Reset(m_handle);

		//std::cout << "m_Reset: m_handle = " << m_handle << std::endl;
	}

	void AddMeasure(__int64 captureTime, const double* val3d)
	{
		m_AddMeasure(m_handle, captureTime, val3d);

		//std::cout << "m_AddMeasure: m_handle = " << m_handle << ", val3d = " << val3d[0] << std::endl;
	}

	void MeasureFrequency(double freq, int frameInd, bool showMixture)
	{
		m_MeasureFrequency(m_handle, freq, frameInd, showMixture);

		//std::cout << "m_MeasureFrequency: m_handle = " << m_handle << ", freq = " << freq << std::endl;
	}
	///
	void GetFrequency(FrequencyResults* freqResults)
	{
		m_GetFrequency(m_handle, freqResults);

		//std::cout << "m_GetFreq: m_handle = " << m_handle << ", freq = " << freq << std::endl;
	}
	///
	int RemainingMeasurements()
	{
		int count = 0;
		m_RemainingMeasurements(m_handle, &count);

		//std::cout << "m_RemainingMeasurements: m_handle = " << m_handle << ", count = " << count << std::endl;

		return count;
	}
	///
	bool GetSignal(SignalInfo* signalInfo)
	{
		if (m_GetSignal(m_handle, signalInfo) == 0)
		{
			return true;
		}
		return false;
	}

private:
	intptr_t m_handle = 0;
	std::string m_dllName = "signal0.dll";

	boost::dll::shared_library m_lib;

	typedef intptr_t(__cdecl CreatePlugin_t)(const InputParams*);
	typedef int(__cdecl DestroyPlugin_t)(intptr_t);
	typedef int(__cdecl Reset_t)(intptr_t);
	typedef int(__cdecl AddMeasure_t)(intptr_t, __int64, const double*);
	typedef int(__cdecl MeasureFrequency_t)(intptr_t, double, int, bool);
	typedef int(__cdecl GetFrequency_t)(intptr_t, FrequencyResults*);
	typedef int(__cdecl RemainingMeasurements_t)(intptr_t, int*);
	typedef int(__cdecl GetSignal_t)(intptr_t, SignalInfo*);

	boost::function<CreatePlugin_t> m_CreatePlugin;
	boost::function<DestroyPlugin_t> m_DestroyPlugin;
	boost::function<Reset_t> m_Reset;
	boost::function<AddMeasure_t> m_AddMeasure;
	boost::function<MeasureFrequency_t> m_MeasureFrequency;
	boost::function<GetFrequency_t> m_GetFrequency;
	boost::function<RemainingMeasurements_t> m_RemainingMeasurements;
	boost::function<GetSignal_t> m_GetSignal;
};