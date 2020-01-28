#include "../plugin.h"
#include "SignalProcessorColor.h"
#include <memory>
#include <vector>

///
intptr_t PLUGIN_FTYPE CreatePlugin(const InputParams* inputParams)
{
    SignalProcessorColor* signalProcess = new SignalProcessorColor(inputParams->framesCount,
		(MeasureSettings::RGBFilters)inputParams->filterType,
		inputParams->signalNormalization,
		inputParams->gauss_def_var,
		inputParams->gauss_min_var,
		inputParams->gauss_max_var,
		inputParams->gauss_eps,
		inputParams->gauss_update_alpha,
		inputParams->gauss_proc_alpha,
		inputParams->gauss_proc_weight_thresh,
		inputParams->retExpFreq);
    return reinterpret_cast<intptr_t>(signalProcess);
}

///
int PLUGIN_FTYPE DestroyPlugin(intptr_t handle)
{
    SignalProcessorColor* signalProcess = reinterpret_cast<SignalProcessorColor*>(handle);
    if (signalProcess == nullptr)
    {
        return -1;
    }
    delete signalProcess;
    return 0;
}

///
/// \brief Reset
///
int PLUGIN_FTYPE Reset(intptr_t handle)
{
	SignalProcessorColor* signalProcess = reinterpret_cast<SignalProcessorColor*>(handle);
	if (signalProcess == nullptr)
	{
		return -1;
	}
	signalProcess->Reset();
	return 0;
}

///
/// \brief MeasureFrequency
///
int PLUGIN_FTYPE MeasureFrequency(intptr_t handle, double freq, int frameInd, bool showMixture)
{
	SignalProcessorColor* signalProcess = reinterpret_cast<SignalProcessorColor*>(handle);
	if (signalProcess == nullptr)
	{
		return -1;
	}
	signalProcess->MeasureFrequency(freq, frameInd, showMixture);
	return 0;
}

///
/// \brief Добавление измерения
/// \param captureTime
/// \param val - pointer to the 3 color values
///
int PLUGIN_FTYPE AddMeasure(intptr_t handle, __int64 captureTime, const double* val3d)
{
	SignalProcessorColor* signalProcess = reinterpret_cast<SignalProcessorColor*>(handle);
	if (signalProcess == nullptr)
	{
		return -1;
	}
	signalProcess->AddMeasure(captureTime, SignalProcessorColor::ClVal_t(val3d[0], val3d[1], val3d[2]));
	return 0;
}

///
/// \brief GetFrequency
/// \param handle
/// \param freqResults
/// \return
///
int PLUGIN_FTYPE GetFrequency(intptr_t handle, FrequencyResults* freqResults)
{
	SignalProcessorColor* signalProcess = reinterpret_cast<SignalProcessorColor*>(handle);
	if (signalProcess == nullptr)
	{
		return -1;
	}
	signalProcess->GetFrequency(freqResults);
	return 0;
}

///
/// \brief RemainingMeasurements
/// \return
///
int PLUGIN_FTYPE RemainingMeasurements(intptr_t handle, int* framesCount)
{
	SignalProcessorColor* signalProcess = reinterpret_cast<SignalProcessorColor*>(handle);
	if (signalProcess == nullptr)
	{
		return -1;
	}
	*framesCount = signalProcess->RemainingMeasurements();
	return 0;
}

///
/// \brief GetSignal
/// \return
///
int PLUGIN_FTYPE GetSignal(intptr_t handle, SignalInfo* signalInfo)
{
	SignalProcessorColor* signalProcess = reinterpret_cast<SignalProcessorColor*>(handle);
	if (signalProcess == nullptr)
	{
		return -1;
	}
	signalProcess->GetSignal(signalInfo);
	return 0;
}
