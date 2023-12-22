#pragma once
#include <vector>
#include <opencv2/imgproc.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

struct NormalizedRect
{
    float x_center = 0;
    float y_center = 0;
    float height = 0;
    float width = 0;
    float rotation = 0;
};

struct Detection
{
    float xmin = 0;
    float xmax = 0;
    float ymin = 0;
    float ymax = 0;
    float score = 0;

    float width = 0;
    float height = 0;

    float rotation = 0;
    std::array<std::pair<float, float>, 6> keypoints;

    float Height() const { return (ymax - ymin); }

    float Width() const { return (xmax - xmin); }

    float Area() const { return Height() * Width(); }

    Detection Intersection(const Detection other) const
    {
        Detection intersect;
        intersect.xmin = std::max(xmin, other.xmin);
        intersect.ymin = std::max(ymin, other.ymin);
        intersect.xmax = std::min(xmax, other.xmax);
        intersect.ymax = std::min(ymax, other.ymax);
        if (intersect.xmin > intersect.xmax || intersect.ymin > intersect.ymax)
            return Detection();
        else
            return intersect;
    }
};

using Detections = std::vector<Detection>;

class FaceDetector
{
public:
    FaceDetector(std::string modelPath);
    ~FaceDetector() = default;

    std::vector<NormalizedRect> Process(const cv::Mat& cvImage);

private:
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::vector<NormalizedRect> anchors;

    void GenerateAnchors();
    void PreProcess(const cv::Mat& originalImage, cv::Mat& transformed, std::array<float, 16>& transformMatrix);
    void PostProcess(Detections& resultingBoxes, const cv::Mat& originalImage);

    float GetScale(int index, size_t total);
    Detection DecodeBox(const float* const rawBox, const NormalizedRect& anchor);
    float SigmoidScore(float rawScore);
    Detections FilterBoxes(const Detections& input);
    Detections ExtractCandidates(Detections& remainingInput);
    float CalculateOverlap(const Detection& a, const Detection& b);
    Detection GetWeightedUnion(const Detections& candidates);
    inline float NormalizeRadians(float angle);
    void GetTransformMatrix(const cv::RotatedRect rect, const int height, const int width, std::array<float, 16>& transformMatrix);
    void ProjectDetections(const std::array<float, 16>& transformMatrix, Detections& input);
};
