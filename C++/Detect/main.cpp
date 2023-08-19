#include "opencv2/opencv.hpp"
#include "yolov8-detect.hpp"
#include <chrono>
using namespace chrono;
int main()
{
    // cuda:0
    //cudaSetDevice(0);
    int num_class = 80;
    string engine_file_path = "yolov8s-detect.engine";
    Yolov8Detect yolov8_detect(engine_file_path, num_class);
    yolov8_detect.make_pipe();


    Mat  image;
    Size size = Size{ 640, 640 };
    int topk = 20;
    float score_thres = 0.4f;
    float iou_thres = 0.65f;

    vector<Object> objs;

    VideoCapture cap(0);

    while (cap.read(image)) {
        flip(image, image, 1);
        yolov8_detect.load_from_mat(image, size);
        auto start = system_clock::now();
        yolov8_detect.infer();
        auto end = system_clock::now();
        yolov8_detect.postprocess(objs, score_thres, iou_thres, topk);
        yolov8_detect.draw_objects(image, objs);
        auto tc = duration_cast<microseconds>(end - start).count() / 1000.0;
        printf("cost %2.4lf ms\n", tc);
        imshow("result", image);
        if (waitKey(1) == 'q') break;
    }
    destroyAllWindows();
    return 0;
}