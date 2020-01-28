#pragma once

#include <cstddef>
#include <stdint.h>

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
//#  ifdef signal_EXPORTS
#    define PLUGIN_EXPORTS __declspec(dllexport)
//#  else
//#    define PLUGIN_EXPORTS __declspec(dllimport)
//#  endif
#  define PLUGIN_FTYPE __cdecl
#elif defined __GNUC__ && __GNUC__ >= 4
#  define PLUGIN_EXPORTS __attribute__ ((visibility ("default")))
#  define PLUGIN_FTYPE
#else
#  define PLUGIN_EXPORTS
#  define PLUGIN_FTYPE
#endif

#pragma pack(push, 1)

///
/// Информация о сигнале, спектре и других параметрах, которые будут отображаться пользователю
/// Все параметры надо просто заполнить или сделать нулями, если данных нет
/// Память под массивы выделять не надо, все значения должна хранить в себе dll, а передавать просто указатели на них
struct SignalInfo
{
	/// История сигнала за последние N кадров, например за framesCount, передаваемое в CreatePlugin
	/// Заполняется 1, 2 или 3 указателя в зависимости от того, какая история есть.
	/// Если используется PCA, то заполняется только один указатель, если ICA, то производятся измерения на всех 3-х каналах и отображатьсЯ будет 3 разных сигнала
	double* m_signal[3];
	/// Реальный размер каждого из массивов в m_signal
	int m_signalSize[3];
	/// Промежуток времени в миллисекундах между соседними отсчётами в m_signal
	double m_deltaTime;
	/// Мощность спектра: 1, 2 или 3 спектра
	double* m_spectrum[3];
	/// Реальный размер каждого из массивов в m_spectrum
	int m_spectrumSize[3];
    /// Целые значения с частотами, соответствующими столбикам спекта (значения выводятся под ними)
	int* m_freqValues[3];
	/// Реальный размер каждого из массивов в m_freqValues
	int m_valuesSize[3];
	/// Индексы от m_fromInd и до m_toInd в массиве спектра, для которых заполнены m_freqValues
	int m_fromInd[3];
	int m_toInd[3];
};

///
struct InputParams
{
	size_t framesCount;               // размер окна сигнала, по которому вычисляется пульс
	int filterType;                   // Тип фильтра, который необходимо применять к RGB сигналу: PCA, ICA или брать только Green канал
	bool signalNormalization;         // использовать или нет нормализацию сигнала
	float gauss_def_var;
	float gauss_min_var;
	float gauss_max_var;
	float gauss_eps;
	float gauss_update_alpha;
	float gauss_proc_alpha;
	float gauss_proc_weight_thresh;
	bool retExpFreq;
	float fps;
};

///
struct FrequencyResults
{
	double smootFreq = 0;    // Smoothed heart rate value for user
	double freq = 0;         // Heart rate on current frame
	double minFreq = 0;      // Minimal value of the heart rate
	double maxFreq = 0;      // Maximum value of the heart rate
	double snr = 0;          // SNR value
	double averageCardiointerval = 0; // Average cardio interval in ms
	double currentCardiointerval = 0; // Current cardio interval in ms
};

#pragma pack(pop)

extern "C"
{
	///
	/// Создание инстанса с вычислением пульса.
	/// Инстанс хранит в себе все необходимы данные: сигнал, частоты и т.д., которые передаются ему на вход функцией AddMeasure
	/// framesCount - размер окна сигнала, по которому вычисляется пульс
	/// Тип фильтра, который необходимо применять к RGB сигналу: PCA, ICA или брать только Green канал
	/// signalNormalization - использовать или нет нормализацию сигнала
	/// params - список дополнительных параметров, сейчас это настройки для гаусовских процессов
	/// paramsCount - количество настроек
	/// \return pointer to the instance
	///
    PLUGIN_EXPORTS intptr_t PLUGIN_FTYPE CreatePlugin(const InputParams* inputParams);
    ///
	/// Удаление инстанса
	/// \return 0 if succed and another if fails
	///
	PLUGIN_EXPORTS int PLUGIN_FTYPE DestroyPlugin(intptr_t handle);

    ///
    /// \brief Reset
	/// Сброс данных и истории
	/// \return 0 if succed and another if fails
    ///
    PLUGIN_EXPORTS int PLUGIN_FTYPE Reset(intptr_t handle);

	///
	/// \brief MeasureFrequency
	/// Произвести вычисления над накопленной историей измерений
	/// freq - текущая частота, относительно которой производятся измерения. Для видео она равна 1000. Для web и сетевых камер может меняться
	/// frameInd и showMixture - параметры для гаусовских процессов
	/// \return 0 if succed and another if fails
	///
	PLUGIN_EXPORTS int PLUGIN_FTYPE MeasureFrequency(intptr_t handle, double freq, int frameInd, bool showMixture);

    ///
    /// \brief Добавление нового измерения
    /// \param captureTime
    /// \param val - pointer to the 3 color values
	/// \return 0 if succed and another if fails
    ///
    PLUGIN_EXPORTS int PLUGIN_FTYPE AddMeasure(intptr_t handle, __int64 captureTime, const double* val3d);

    ///
    /// \brief GetFrequency - получение значения частоты и других дополнительных величин, которые можно вывести пользователю
	/// \param freqResults - значение частоты и других величин
	/// \return 0 if succed and another if fails
    ///
    PLUGIN_EXPORTS int PLUGIN_FTYPE GetFrequency(intptr_t handle, FrequencyResults* freqResults);

    ///
    /// \brief RemainingMeasurements
	/// Получение числа кадров, которые осталось накопить, чтобы начать выводить пользователю частоту
    /// \return 0 if succed and another if fails
    ///
    PLUGIN_EXPORTS int PLUGIN_FTYPE RemainingMeasurements(intptr_t handle, int* framesCount);

	///
	/// \brief GetSignal
	/// Получение сигнала, спектра и других значений для отображения их на графиках
	/// \return 0 if succed and another if fails
	///
	PLUGIN_EXPORTS int PLUGIN_FTYPE GetSignal(intptr_t handle, SignalInfo* signalInfo);
}
