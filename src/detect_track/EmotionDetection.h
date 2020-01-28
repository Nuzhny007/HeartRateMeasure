#pragma once
#include <opencv2/opencv.hpp>

class EmotionRecognition
{
public:
	enum Emotions
	{
		Neutral,
		Happy,
		Sad,
		Surprise,
		Anger
	};

	EmotionRecognition() {}
	virtual ~EmotionRecognition() {}

	virtual Emotions Recognition(const cv::Mat& face) = 0;

	Emotions Ind2Emo(int ind)
	{
		switch (ind)
		{
		case 0:
			return Neutral;
		case 1:
			return Happy;
		case 2:
			return Sad;
		case 3:
			return Surprise;
		case 4:
			return Anger;
		default:
			return Neutral;
		}
	}
};

EmotionRecognition* CreateRecognizer(const std::string& appDirPath);
