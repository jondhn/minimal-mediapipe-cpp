#include "face_detector.h"
#include <tensorflow/lite/kernels/register.h>

constexpr auto M_PI = (3.14159265358979323846);

const float minScale = 0.1484375f;
const float maxScale = 0.75f;
const int strides[4] = { 8, 16, 16, 16 };
const float offset = 0.5f;
const float xScale = 128.0f;
const float yScale = 128.0f;
const float hScale = 128.0f;
const float wScale = 128.0f;
const float minScoreThresh = 0.5f;
const float scoreClippingThresh = 100.f;
const float minSuppressionThreshold = 0.3f;
const int inputWidth = 128;
const int inputHeight = 128;
const int inputChannels = 3;

FaceDetector::FaceDetector(std::string modelPath)
{
    model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (model == nullptr)
        throw std::invalid_argument("loading face detection model failed");

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&interpreter);
    if (interpreter == nullptr)
        throw std::invalid_argument("face detection: building interpreter failed");
    if (interpreter->AllocateTensors() != kTfLiteOk)
        throw std::invalid_argument("face detection: allocating Tensors failed");
    GenerateAnchors();
}

float FaceDetector::GetScale(int index, size_t total)
{
    if (total == 1)
        return (minScale + maxScale) / 2.0f;

    return minScale + (maxScale - minScale) * (index * 1.0f) / (total - 1.0f);
}

Detection FaceDetector::DecodeBox(const float* const rawBox, const NormalizedRect& anchor)
{
    const auto centerX = rawBox[0];
    const auto centerY = rawBox[1];
    const auto width = rawBox[2];
    const auto height = rawBox[3];

    float scaled_centerX = centerX / xScale * anchor.width + anchor.x_center;
    float scaled_centerY = centerY / yScale * anchor.height + anchor.y_center;
    float scaled_height = height / hScale * anchor.height;
    float scaled_width = width / wScale * anchor.width;

    Detection detection;
    detection.ymin = scaled_centerY - scaled_height / 2.f;
    detection.xmin = scaled_centerX - scaled_width / 2.f;
    detection.ymax = scaled_centerY + scaled_height / 2.f;
    detection.xmax = scaled_centerX + scaled_width / 2.f;

    return detection;
}

float FaceDetector::SigmoidScore(float rawScore)
{
    const float clippedScore =
        std::min(scoreClippingThresh, std::max(-scoreClippingThresh, rawScore));
    return 1.0f / (1.0f + std::exp(-clippedScore));
}

Detections FaceDetector::FilterBoxes(const Detections& input)
{
    if (input.size() == 0)
        throw std::invalid_argument("no face found in mediapipe face detection");

    std::vector<Detection> output;

    Detections remainingInput;
    remainingInput.assign(input.begin(), input.end());
    auto SortByScore = [](const Detection& input0, const Detection& input1)
    {
        return (input0.score > input1.score);
    };

    std::sort(remainingInput.begin(), remainingInput.end(), SortByScore);

    while (!remainingInput.empty())
    {
        const auto candidates = ExtractCandidates(remainingInput);
        output.push_back(GetWeightedUnion(candidates));
    }
    return output;
}

Detections FaceDetector::ExtractCandidates(Detections& remainingInput)
{
    auto& firstBox = remainingInput[0];
    Detections candidates;
    Detections remaining;
    size_t inputItemsCount;
    do
    {
        remaining.clear();
        inputItemsCount = remainingInput.size();

        for (const auto& item : remainingInput)
        {
            auto overlap = 0.0f;
            if (candidates.empty())
                overlap = CalculateOverlap(firstBox, item);
            else
                for (const auto& candidate : candidates)
                    overlap = std::max(overlap, CalculateOverlap(candidate, item));

            if (overlap > minSuppressionThreshold)
            {
                candidates.push_back(item);
            }
            else
            {
                remaining.push_back(item);
            }
        }

        remainingInput.assign(remaining.begin(), remaining.end());
    } while (inputItemsCount != remaining.size());

    return candidates;
}

float FaceDetector::CalculateOverlap(const Detection& a, const Detection& b)
{
    const auto intersectionArea = a.Intersection(b).Area();
    const auto unionArea = a.Area() + b.Area() - intersectionArea;
    return intersectionArea / unionArea;
}

Detection FaceDetector::GetWeightedUnion(const Detections& candidates)
{
    Detection output;
    auto totalScore = .0f;
    int num_relevant_keypoints = 2;
    std::vector<float> keypoints(num_relevant_keypoints * 2);
    for (const auto& item : candidates)
    {
        totalScore += item.score;
        output.xmin += item.xmin * item.score;
        output.ymin += item.ymin * item.score;
        output.xmax += (item.xmin + item.Width()) * item.score;
        output.ymax += (item.ymin + item.Height()) * item.score;

        for (int i = 0; i < num_relevant_keypoints; ++i)
        {
            keypoints[i * 2] += item.keypoints[i].first * item.score;
            keypoints[i * 2 + 1] += item.keypoints[i].second * item.score;
        }
    }
    output.xmin /= totalScore;
    output.ymin /= totalScore;
    output.width = (output.xmax / totalScore) - output.xmin;
    output.height = (output.ymax / totalScore) - output.ymin;

    for (int i = 0; i < num_relevant_keypoints; ++i)
    {
        output.keypoints[i].first = keypoints[i * 2] / totalScore;
        output.keypoints[i].second = keypoints[i * 2 + 1] / totalScore;
    }

    return output;
}

inline float FaceDetector::NormalizeRadians(float angle)
{
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

void FaceDetector::GenerateAnchors()
{
    anchors.clear();
    int layerId = 0;
    while (layerId < sizeof(strides))
    {
        std::vector<float> scales;
        auto firstSameStrideLayer = layerId;

        // bundle all same stride layers
        while (layerId < sizeof(strides) && strides[firstSameStrideLayer] == strides[layerId])
        {
            auto scale = GetScale(layerId, sizeof(strides));
            auto nextScale = GetScale(layerId + 1, sizeof(strides));
            auto interpolated = std::sqrt(scale * nextScale);

            scales.push_back(scale);
            scales.push_back(interpolated);
            layerId++;
        }

        const auto featureHeight = static_cast<int>(
            std::ceil(1.0f * 128 / strides[firstSameStrideLayer]));
        const auto featureWidth = static_cast<int>(
            std::ceil(1.0f * 128 / strides[firstSameStrideLayer]));

        for (int y = 0; y < featureHeight; y++)
        {
            const auto centerY = (y + offset) / featureHeight;
            for (int x = 0; x < featureWidth; x++)
            {
                const auto centerX = (x + offset) / featureWidth;
                for (auto& scale : scales)
                {

                    auto& anchor = anchors.emplace_back();

                    anchor.x_center = centerX;
                    anchor.y_center = centerY;
                    anchor.width = anchor.height = 1.0f;
                }
            }
        }
    }
}

// function GetRotatedSubRectToRectTransformMatrix in image_to_tensor_utils.cc
void FaceDetector::GetTransformMatrix(const cv::RotatedRect rect, const int height, const int width, std::array<float, 16>& transformMatrix)
{
    // The resulting matrix is multiplication of below commented out matrices:
    //   post_scale_matrix
    //     * translate_matrix
    //     * rotate_matrix
    //     * flip_matrix
    //     * scale_matrix
    //     * initial_translate_matrix

    // Matrix to convert X,Y to [-0.5, 0.5] range "initial_translate_matrix"
    // { 1.0f,  0.0f, 0.0f, -0.5f}
    // { 0.0f,  1.0f, 0.0f, -0.5f}
    // { 0.0f,  0.0f, 1.0f,  0.0f}
    // { 0.0f,  0.0f, 0.0f,  1.0f}
    const float a = rect.size.width;
    const float b = rect.size.height;

    const float c = std::cos(rect.angle);
    const float d = std::sin(rect.angle);
    // Matrix to do rotation around Z axis "rotate_matrix"
    // {    c,   -d, 0.0f, 0.0f}
    // {    d,    c, 0.0f, 0.0f}
    // { 0.0f, 0.0f, 1.0f, 0.0f}
    // { 0.0f, 0.0f, 0.0f, 1.0f}

    const float e = rect.center.x;
    const float f = rect.center.y;
    // Matrix to do X,Y translation of sub rect within parent rect
    // "translate_matrix"
    // {1.0f, 0.0f, 0.0f, e   }
    // {0.0f, 1.0f, 0.0f, f   }
    // {0.0f, 0.0f, 1.0f, 0.0f}
    // {0.0f, 0.0f, 0.0f, 1.0f}

    const float g = 1.0f / width;
    const float h = 1.0f / height;
    // Matrix to scale X,Y,Z to [0.0, 1.0] range "post_scale_matrix"
    // {g,    0.0f, 0.0f, 0.0f}
    // {0.0f, h,    0.0f, 0.0f}
    // {0.0f, 0.0f,    g, 0.0f}
    // {0.0f, 0.0f, 0.0f, 1.0f}

    // row 1
    transformMatrix[0] = a * c * g;
    transformMatrix[1] = -b * d * g;
    transformMatrix[2] = 0.0f;
    transformMatrix[3] = (-0.5f * a * c + 0.5f * b * d + e) * g;

    // row 2
    transformMatrix[4] = a * d * h;
    transformMatrix[5] = b * c * h;
    transformMatrix[6] = 0.0f;
    transformMatrix[7] = (-0.5f * b * c - 0.5f * a * d + f) * h;

    // row 3
    transformMatrix[8] = 0.0f;
    transformMatrix[9] = 0.0f;
    transformMatrix[10] = a * g;
    transformMatrix[11] = 0.0f;

    // row 4
    transformMatrix[12] = 0.0f;
    transformMatrix[13] = 0.0f;
    transformMatrix[14] = 0.0f;
    transformMatrix[15] = 1.0f;
}

void FaceDetector::ProjectDetections(const std::array<float, 16>& transformMatrix, Detections& input)
{
    auto project_fn = [transformMatrix](const cv::Point2f& p) -> cv::Point2f {
        return { p.x * transformMatrix[0] + p.y * transformMatrix[1] + transformMatrix[3],
                p.x * transformMatrix[4] + p.y * transformMatrix[5] + transformMatrix[7] };
    };
    for (auto& detection : input)
    {

        for (int i = 0; i < detection.keypoints.size(); ++i)
        {
            auto& kp = detection.keypoints[i];
            const auto point = project_fn({ kp.first, kp.second });
            kp.first = point.x;
            kp.second = point.y;
        }

        const float xmin = detection.xmin;
        const float ymin = detection.ymin;
        const float width = detection.width;
        const float height = detection.height;
        // a) Define and project box points.
        std::array<cv::Point2f, 4> box_coordinates = {
            cv::Point2f{xmin, ymin}, cv::Point2f{xmin + width, ymin},
            cv::Point2f{xmin + width, ymin + height}, cv::Point2f{xmin, ymin + height} };
        std::transform(box_coordinates.begin(), box_coordinates.end(),
            box_coordinates.begin(), project_fn);
        // b) Find new left top and right bottom points for a box which encompases
        //    non-projected (rotated) box.
        constexpr float kFloatMax = std::numeric_limits<float>::max();
        constexpr float kFloatMin = std::numeric_limits<float>::lowest();
        cv::Point2f left_top = { kFloatMax, kFloatMax };
        cv::Point2f right_bottom = { kFloatMin, kFloatMin };
        std::for_each(box_coordinates.begin(), box_coordinates.end(),
            [&left_top, &right_bottom](const cv::Point2f& p) {
            left_top.x = std::min(left_top.x, p.x);
            left_top.y = std::min(left_top.y, p.y);
            right_bottom.x = std::max(right_bottom.x, p.x);
            right_bottom.y = std::max(right_bottom.y, p.y);
        });
        detection.xmin = left_top.x;
        detection.ymin = left_top.y;
        detection.width = right_bottom.x - left_top.x;
        detection.height = right_bottom.y - left_top.y;
    }
}

std::vector<NormalizedRect> FaceDetector::Process(const cv::Mat& cvImage)
{
    cv::Mat transformed;
    std::array<float, 16> transformMatrix;
    PreProcess(cvImage, transformed, transformMatrix);
    float* input = interpreter->typed_input_tensor<float>(0);
    memcpy(
        input,
        transformed.ptr<float>(0),
        inputWidth * inputHeight * inputChannels * sizeof(float));
    interpreter->Invoke();

    Detections resultingBoxes;
    PostProcess(resultingBoxes, cvImage);

    auto filteredBoxes = FilterBoxes(resultingBoxes);
    ProjectDetections(transformMatrix, filteredBoxes);
    // filtered boxes to bounding boxes
    std::vector<NormalizedRect> output;
    for (Detection& detection : filteredBoxes)
    {
        // calculate box rotation
        const float x0 = detection.keypoints[0].first * cvImage.cols;
        const float y0 = detection.keypoints[0].second * cvImage.rows;
        const float x1 = detection.keypoints[1].first * cvImage.cols;
        const float y1 = detection.keypoints[1].second * cvImage.rows;

        NormalizedRect rect;
        float scale = 1.5f;
        rect.x_center = detection.xmin + detection.width / 2;
        rect.y_center = detection.ymin + detection.height / 2;
        rect.width = detection.width * scale;
        rect.height = detection.height * scale;
        rect.rotation = NormalizeRadians(-std::atan2(-(y1 - y0), x1 - x0));

        const float long_side = std::max(rect.width * cvImage.cols, rect.height * cvImage.rows);
        rect.width = long_side / cvImage.cols;
        rect.height = long_side / cvImage.rows;
        output.push_back(rect);
    }
    return output;
}

void FaceDetector::PreProcess(const cv::Mat& originalImage, cv::Mat& transformed, std::array<float, 16>& transformMatrix)
{
    float cx = originalImage.cols * 0.5f;
    float cy = originalImage.rows * 0.5f;
    float h = originalImage.rows;
    float w = originalImage.cols;
    float rotation = 0;

    const float dst_width = 128;
    const float dst_height = 128;

    const float tensor_aspect_ratio = static_cast<float>(dst_height) / dst_width;
    const float roi_aspect_ratio = (float)h / (float)w;

    float vertical_padding = 0.0f;
    float horizontal_padding = 0.0f;
    float new_width;
    float new_height;
    if (tensor_aspect_ratio > roi_aspect_ratio)
    {
        new_width = w;
        new_height = w * tensor_aspect_ratio;
    }
    else
    {
        new_width = h / tensor_aspect_ratio;
        new_height = h;
    }

    const cv::RotatedRect rotated_rect(cv::Point2f(cx, cy),
        cv::Size2f(new_width, new_height),
        rotation * 180.f / M_PI);
    GetTransformMatrix(rotated_rect, originalImage.rows, originalImage.cols, transformMatrix);
    cv::Mat src_points;
    cv::boxPoints(rotated_rect, src_points);

    float dst_corners[8] = { 0.0f,      dst_height,
                            0.0f,      0.0f,
                            dst_width, 0.0f,
                            dst_width, dst_height };

    cv::Mat dst_points = cv::Mat(4, 2, CV_32F, dst_corners);
    cv::Mat projection_matrix = cv::getPerspectiveTransform(src_points, dst_points);
    cv::warpPerspective(originalImage, transformed, projection_matrix,
        cv::Size(dst_width, dst_height), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    // normalize image
    constexpr float kInputImageRangeMin = 0.0f;
    constexpr float kInputImageRangeMax = 255.0f;
    int rangeMax = 1;
    int rangeMin = -1;
    const float scale = (rangeMax - rangeMin) / (kInputImageRangeMax - kInputImageRangeMin);
    const float offset = rangeMin - kInputImageRangeMin * scale;
    transformed.convertTo(transformed, 21, scale, offset);
}

void FaceDetector::PostProcess(Detections& resultingBoxes, const cv::Mat& originalImage)
{
    float* regressors = interpreter->typed_output_tensor<float>(0);
    float* score = interpreter->typed_output_tensor<float>(1);

    const int keypoint_coord_offset = 4;
    const int num_values_per_keypoint = 2;
    const int num_relevant_keypoints = 2;
    const int num_coordinates = 16;
    const int num_boxes = 896;

    for (int i = 0; i < num_boxes; i++)
    {
        float sigmoid = SigmoidScore(score[i]);

        if (sigmoid >= minScoreThresh)
        {
            Detection detection = DecodeBox(regressors + i * num_coordinates, anchors[i]);
            detection.score = sigmoid;

            // calculate keypoints
            int box_offset = i * num_coordinates;
            for (int k = 0; k < num_relevant_keypoints; ++k)
            {
                const int offset = box_offset + keypoint_coord_offset + k * num_values_per_keypoint;

                float keypoint_x = regressors[offset];
                float keypoint_y = regressors[offset + 1];

                detection.keypoints[k].first = keypoint_x / xScale * anchors[i].width + anchors[i].x_center;
                detection.keypoints[k].second = keypoint_y / yScale * anchors[i].height + anchors[i].y_center;
            }
            resultingBoxes.emplace_back(detection);
        }
    }
}
