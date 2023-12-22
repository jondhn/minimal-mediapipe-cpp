#include "FaceLib/landmark_extractor.h"
#include <opencv2/highgui.hpp>
#include <iostream>

int main()
{
	std::string imageFile = "../data/images/image.jpg";
	cv::Mat image = cv::imread(imageFile);
	
	std::string detectionModelPath = "../data/models/face_detection_short_range.tflite";
	std::string landmarksModelPath = "../data/models/face_landmarks.tflite";
	LandmarkExtractor extractor = LandmarkExtractor(detectionModelPath, landmarksModelPath);

	try
	{
		std::vector<cv::Point2i> landmarks = extractor.Process(image);
		for (auto& lm : landmarks)
		{
			cv::circle(image, lm, 3, cv::Scalar(204, 102, 0), -1);
		}
		cv::imshow("Face Landmarks", image);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
	catch (std::exception& ex)
	{
		std::cout << ex.what() << std::endl;
	}
	return 0;
}