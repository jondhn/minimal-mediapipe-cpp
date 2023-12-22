#pragma once
#include "face_detector.h"
#include <opencv2/imgproc.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

class LandmarkExtractor
{
public:
    LandmarkExtractor(std::string detectionModelPath, std::string landmarksModelPath);
    ~LandmarkExtractor() = default;

    std::vector<cv::Point2i> Process(const cv::Mat& cvImage);

private:
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::FlatBufferModel> model;

    std::shared_ptr<FaceDetector> face_detector{ nullptr };
    float detection_threshold = 0.5;

    void PreProcess(const cv::Mat& originalImage, cv::Mat& transformed, const NormalizedRect rect);
    void PostProcess(const cv::Mat& transformed, const NormalizedRect& box, std::vector<cv::Point2i>& landmarks);
};
