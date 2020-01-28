#include "../plugin.h"
#include <memory>
#include <vector>
#include "../../common/common.h"
#include "VPGSignalProcessor.h"

///
intptr_t PLUGIN_FTYPE CreatePlugin(const InputParams* inputParams)
{
	VPGSignalProcessor* signalProcess = new VPGSignalProcessor(inputParams->framesCount, inputParams->fps, inputParams->retExpFreq);
    return reinterpret_cast<intptr_t>(signalProcess);
}

///
int PLUGIN_FTYPE DestroyPlugin(intptr_t handle)
{
    VPGSignalProcessor* signalProcess = reinterpret_cast<VPGSignalProcessor*>(handle);
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
	VPGSignalProcessor* signalProcess = reinterpret_cast<VPGSignalProcessor*>(handle);
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
	VPGSignalProcessor* signalProcess = reinterpret_cast<VPGSignalProcessor*>(handle);
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
	VPGSignalProcessor* signalProcess = reinterpret_cast<VPGSignalProcessor*>(handle);
	if (signalProcess == nullptr)
	{
		return -1;
	}
	signalProcess->AddMeasure(captureTime, VPGSignalProcessor::ClVal_t(val3d[0], val3d[1], val3d[2]));
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
	VPGSignalProcessor* signalProcess = reinterpret_cast<VPGSignalProcessor*>(handle);
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
	VPGSignalProcessor* signalProcess = reinterpret_cast<VPGSignalProcessor*>(handle);
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
	VPGSignalProcessor* signalProcess = reinterpret_cast<VPGSignalProcessor*>(handle);
	if (signalProcess == nullptr)
	{
		return -1;
	}
	signalProcess->GetSignal(signalInfo);
	return 0;
}
