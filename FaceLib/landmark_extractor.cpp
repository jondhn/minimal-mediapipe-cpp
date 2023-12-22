#include "landmark_extractor.h"
#include <tensorflow/lite/kernels/register.h>

constexpr auto M_PI = (3.14159265358979323846);

const int WIDTH = 192;
const int HEIGHT = 192;
const int CHANNEL = 3;
const int OUTDIM = 1404;

const int NUM_LANDMARKS = 468;
const int NUM_ATTENTION_LANDMARKS = 478;
const int NUM_LIP_LANDMARKS = 80;
const int NUM_EYE_LANDMARKS = 71;
const int NUM_IRIS_LANDMARKS = 5;

const int lip_idx[NUM_LIP_LANDMARKS] = {
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        // Upper outer(excluding corners).
        185, 40, 39, 37, 0, 267, 269, 270, 409,
        // Lower inner.
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        // Upper inner(excluding corners).
        191, 80, 81, 82, 13, 312, 311, 310, 415,
        // Lower semi - outer.
        76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306,
        // Upper semi - outer(excluding corners).
        184, 74, 73, 72, 11, 302, 303, 304, 408,
        // Lower semi - inner.
        62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292,
        // Upper semi - inner(excluding corners).
        183, 42, 41, 38, 12, 268, 271, 272, 407
};
// left eye attention
const int left_eye_idx[NUM_EYE_LANDMARKS] = {
    // Lower contour.
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    // upper contour(excluding corners).
    246, 161, 160, 159, 158, 157, 173,
    // Halo x2 lower contour.
    130, 25, 110, 24, 23, 22, 26, 112, 243,
    // Halo x2 upper contour(excluding corners).
    247, 30, 29, 27, 28, 56, 190,
    // Halo x3 lower contour.
    226, 31, 228, 229, 230, 231, 232, 233, 244,
    // Halo x3 upper contour(excluding corners).
    113, 225, 224, 223, 222, 221, 189,
    // Halo x4 upper contour(no lower because of mesh structure) or
    // eyebrow inner contour.
    35, 124, 46, 53, 52, 65,
    // Halo x5 lower contour.
    143, 111, 117, 118, 119, 120, 121, 128, 245,
    // Halo x5 upper contour(excluding corners) or eyebrow outer contour.
    156, 70, 63, 105, 66, 107, 55, 193
};
// right eye attention
const int right_eye_idx[NUM_EYE_LANDMARKS] = {
    // Lower contour.
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    // Upper contour(excluding corners).
    466, 388, 387, 386, 385, 384, 398,
    // Halo x2 lower contour.
    359, 255, 339, 254, 253, 252, 256, 341, 463,
    // Halo x2 upper contour(excluding corners).
    467, 260, 259, 257, 258, 286, 414,
    // Halo x3 lower contour.
    446, 261, 448, 449, 450, 451, 452, 453, 464,
    // Halo x3 upper contour(excluding corners).
    342, 445, 444, 443, 442, 441, 413,
    // Halo x4 upper contour(no lower because of mesh structure) or
    // eyebrow inner contour.
    265, 353, 276, 283, 282, 295,
    // Halo x5 lower contour.
    372, 340, 346, 347, 348, 349, 350, 357, 465,
    // Halo x5 upper contour(excluding corners) or eyebrow outer contour.
    383, 300, 293, 334, 296, 336, 285, 417
};
const int left_iris_idx[NUM_IRIS_LANDMARKS] = {
    // Center.
    468,
    // Iris right edge.
    469,
    // Iris top edge.
    470,
    // Iris left edge.
    471,
    // Iris bottom edge.
    472
};
const int right_iris_idx[NUM_IRIS_LANDMARKS] = {
    // Center.
    473,
    // Iris right edge.
    474,
    // Iris top edge.
    475,
    // Iris left edge.
    476,
    // Iris bottom edge.
    477
};

LandmarkExtractor::LandmarkExtractor(std::string detectionModelPath, std::string landmarksModelPath)
{
    face_detector = std::make_shared<FaceDetector>(detectionModelPath);

    model = tflite::FlatBufferModel::BuildFromFile(landmarksModelPath.c_str());
    if (model == nullptr)
        throw std::invalid_argument("loading face landmarks model failed");

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&interpreter);
    if (interpreter == nullptr)
        throw std::invalid_argument("face landmarks: building interpreter failed");
    if (interpreter->AllocateTensors() != kTfLiteOk)
        throw std::invalid_argument("face landmarks: allocating Tensors failed");
}

std::vector<cv::Point2i> LandmarkExtractor::Process(const cv::Mat& image)
{
    cv::Mat rgbImage;
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

    // find biggest face in image
    std::vector<NormalizedRect> out = face_detector->Process(rgbImage);
    if (out.size() < 1)
    {
        throw std::invalid_argument("no face found in image");
    }
    NormalizedRect rect = out[0];

    cv::Mat transformed;
    PreProcess(rgbImage, transformed, rect);

    float* input = interpreter->typed_input_tensor<float>(0);
    memcpy(
        input,
        transformed.ptr<float>(0),
        WIDTH * HEIGHT * CHANNEL * sizeof(float));
    interpreter->Invoke();

    std::vector<cv::Point2i> landmarks;
    PostProcess(rgbImage, rect, landmarks);
    return landmarks;
}

void LandmarkExtractor::PreProcess(const cv::Mat& originalImage, cv::Mat& transformed, const NormalizedRect rect)
{
    float cx = rect.x_center * originalImage.cols;
    float cy = rect.y_center * originalImage.rows;
    float h = rect.height * originalImage.rows;
    float w = rect.width * originalImage.cols;
    float rotation = rect.rotation;

    const cv::RotatedRect rotated_rect(cv::Point2f(cx, cy),
        cv::Size2f(w, h), rotation * 180.f / M_PI);
    cv::Mat src_points;
    cv::boxPoints(rotated_rect, src_points);

    /* clang-format off */
    float dst_corners[8] = { 0.0f,      HEIGHT,
                            0.0f,      0.0f,
                            WIDTH, 0.0f,
                            WIDTH, HEIGHT };
    /* clang-format on */

    cv::Mat dst_points = cv::Mat(4, 2, CV_32F, dst_corners);
    cv::Mat projection_matrix =
        cv::getPerspectiveTransform(src_points, dst_points);
    //cv::Mat transformed;
    cv::warpPerspective(originalImage, transformed, projection_matrix,
        cv::Size(WIDTH, HEIGHT),
        /*flags=*/cv::INTER_LINEAR,
        /*borderMode=*/cv::BORDER_REPLICATE);
    // normalize image
    constexpr float kInputImageRangeMin = 0.0f;
    constexpr float kInputImageRangeMax = 255.0f;
    int rangeMax = 1;
    int rangeMin = 0;
    const float scale =
        (rangeMax - rangeMin) / (kInputImageRangeMax - kInputImageRangeMin);
    const float offset = rangeMin - kInputImageRangeMin * scale;
    transformed.convertTo(transformed, CV_32F, scale, offset);
}

void LandmarkExtractor::PostProcess(const cv::Mat& transformed, const NormalizedRect& box, std::vector<cv::Point2i>& landmarks)
{
    float* faceflag = interpreter->typed_output_tensor<float>(2);
    float* mesh = interpreter->typed_output_tensor<float>(3);
    float* lips = interpreter->typed_output_tensor<float>(5);
    float* left_eye = interpreter->typed_output_tensor<float>(0);
    float* left_iris = interpreter->typed_output_tensor<float>(4);
    float* right_eye = interpreter->typed_output_tensor<float>(1);
    float* right_iris = interpreter->typed_output_tensor<float>(6);

    std::vector<cv::Point2f> output;
    output.insert(output.end(), NUM_ATTENTION_LANDMARKS, cv::Point2f());

    for (size_t i = 0; i < NUM_LANDMARKS; i++)
    {
        output[i].x = mesh[3 * i + 0] / WIDTH;
        output[i].y = mesh[3 * i + 1] / HEIGHT;
    }

    for (int i = 0; i < NUM_EYE_LANDMARKS; i++)
    {
        output[left_eye_idx[i]].x = left_eye[2 * i + 0] / WIDTH;
        output[left_eye_idx[i]].y = left_eye[2 * i + 1] / HEIGHT;
    }
    for (int i = 0; i < NUM_EYE_LANDMARKS; i++)
    {
        output[right_eye_idx[i]].x = right_eye[2 * i + 0] / WIDTH;
        output[right_eye_idx[i]].y = right_eye[2 * i + 1] / HEIGHT;
    }
    for (size_t i = 0; i < NUM_IRIS_LANDMARKS; i++)
    {
        output[NUM_LANDMARKS + i].x = left_iris[2 * i + 0] / WIDTH;
        output[NUM_LANDMARKS + i].y = left_iris[2 * i + 1] / HEIGHT;
    }
    for (size_t i = 0; i < NUM_IRIS_LANDMARKS; i++)
    {
        output[NUM_LANDMARKS + NUM_IRIS_LANDMARKS + i].x = right_iris[2 * i + 0] / WIDTH;
        output[NUM_LANDMARKS + NUM_IRIS_LANDMARKS + i].y = right_iris[2 * i + 1] / HEIGHT;
    }
    for (int i = 0; i < NUM_LIP_LANDMARKS; i++)
    {
        output[lip_idx[i]].x = lips[2 * i + 0] / WIDTH;
        output[lip_idx[i]].y = lips[2 * i + 1] / HEIGHT;
    }
    // scale and rotate landmarks
    landmarks.insert(landmarks.end(), NUM_ATTENTION_LANDMARKS, cv::Point2i());
    for (size_t i = 0; i < NUM_ATTENTION_LANDMARKS; i++)
    {
        float x = output[i].x - 0.5f;
        float y = output[i].y - 0.5f;
        x = std::cos(box.rotation) * x - std::sin(box.rotation) * y;
        y = std::sin(box.rotation) * x + std::cos(box.rotation) * y;
        x = x * box.width + box.x_center;
        y = y * box.height + box.y_center;
        landmarks[i].x = static_cast<int>(std::round(x * transformed.cols));
        landmarks[i].y = static_cast<int>(std::round(y * transformed.rows));
    }
}
