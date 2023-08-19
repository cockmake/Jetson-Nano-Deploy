#include "NvInferPlugin.h"
#include <fstream>
#include <algorithm>
using namespace std;
using namespace cv;
using namespace nvinfer1;
inline float clamp(float val, float min, float max) {
    return val > min ? (val < max ? val : max) : min;
}
inline void cuda_check(cudaError_t error_code) {
    if (error_code != cudaSuccess) {
        printf("CUDA Error:\n");                                                                                   \
            printf("    File:       %s\n", __FILE__);                                                                  \
            printf("    Line:       %d\n", __LINE__);                                                                  \
            printf("    Error code: %d\n", error_code);                                                                \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));                                            \
            exit(1);                                                                                                   \
    }
}
inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}
inline int type_to_size(const nvinfer1::DataType& dataType)
{
    switch (dataType) {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        return 4;
    }
}

namespace pose {
    class Logger : public nvinfer1::ILogger {
    public:
        nvinfer1::ILogger::Severity reportableSeverity;

        explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) :
            reportableSeverity(severity) {};

        void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
        {
            if (severity > reportableSeverity) {
                return;
            }
            switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                cerr << "INTERNAL_ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                cerr << "ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                cerr << "WARNING: ";
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                cerr << "INFO: ";
                break;
            default:
                cerr << "VERBOSE: ";
                break;
            }
            cerr << msg << endl;
        }
    };
    class Binding {
    public:
        size_t size = 1;
        size_t dsize = 1;
        nvinfer1::Dims dims;
        string name;
    };

    struct Object {
        Rect_<float> rect;
        int label = 0;
        float prob = 0.0;
        vector<float> kps;
    };

    struct PreParam {
        float ratio = 1.0f;
        float dw = 0.0f;
        float dh = 0.0f;
        float height = 0;
        float width = 0;
    };
}
using namespace pose;


class Yolov8Pose {
private:
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream = nullptr;
    Logger                       gLogger{ nvinfer1::ILogger::Severity::kINFO };
public:
    int num_bindings, num_inputs = 0, num_outputs = 0;
    vector<Binding> input_bindings, output_bindings;
    vector<void*>   host_ptrs;
    vector<void*>   device_ptrs;
    PreParam pparam;
    const vector<Scalar> KPS_COLORS = {
        {0, 255, 0},
        {0, 255, 0},
        {0, 255, 0},
        {0, 255, 0},
        {0, 255, 0},
        {255, 128, 0},
        {255, 128, 0},
        {255, 128, 0},
        {255, 128, 0},
        {255, 128, 0},
        {255, 128, 0},
        {51, 153, 255},
        {51, 153, 255},
        {51, 153, 255},
        {51, 153, 255},
        {51, 153, 255},
        {51, 153, 255}
    };
    const vector<vector<unsigned int>> SKELETON = {
        {16, 14},
        {14, 12},
        {17, 15},
        {15, 13},
        {12, 13},
        {6, 12},
        {7, 13},
        {6, 7},
        {6, 8},
        {7, 9},
        {8, 10},
        {9, 11},
        {2, 3},
        {1, 2},
        {1, 3},
        {2, 4},
        {3, 5},
        {4, 6},
        {5, 7}
    };
    const vector<Scalar> LIMB_COLORS = {
        {51, 153, 255},
        {51, 153, 255},
        {51, 153, 255},
        {51, 153, 255},
        {255, 51, 255},
        {255, 51, 255},
        {255, 51, 255},
        {255, 128, 0},
        {255, 128, 0},
        {255, 128, 0},
        {255, 128, 0},
        {255, 128, 0},
        {0, 255, 0},
        {0, 255, 0},
        {0, 255, 0},
        {0, 255, 0},
        {0, 255, 0},
        {0, 255, 0},
        {0, 255, 0}
    };
    explicit Yolov8Pose(string& engine_file_path) {
        ifstream file(engine_file_path, ios::binary);
        file.seekg(0, ios::end);
        auto size = file.tellg();
        file.seekg(0, ios::beg);
        char* trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
        initLibNvInferPlugins(&gLogger, "");
        runtime = nvinfer1::createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(trtModelStream, size);
        delete[] trtModelStream;
        context = engine->createExecutionContext();
        cudaStreamCreate(&stream);
        num_bindings = engine->getNbBindings();

        for (int i = 0; i < num_bindings; ++i) {
            Binding binding;
            nvinfer1::Dims dims;
            nvinfer1::DataType dtype = engine->getBindingDataType(i);
            string name = engine->getBindingName(i);
            binding.name = name;
            binding.dsize = type_to_size(dtype);

            bool IsInput = engine->bindingIsInput(i);
            if (IsInput) {
                num_inputs += 1;
                dims = engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
                binding.size = get_size_by_dims(dims);
                binding.dims = dims;
                input_bindings.emplace_back(binding);
                // set max opt shape
                context->setBindingDimensions(i, dims);
            }
            else {
                dims = context->getBindingDimensions(i);
                binding.size = get_size_by_dims(dims);
                binding.dims = dims;
                output_bindings.emplace_back(binding);
                num_outputs += 1;
            }
        }
    }

    ~Yolov8Pose() {
        context->destroy();
        engine->destroy();
        runtime->destroy();
        cudaStreamDestroy(stream);
        for (auto& ptr : device_ptrs) {
            cuda_check(cudaFree(ptr));
        }

        for (auto& ptr : host_ptrs) {
            cuda_check(cudaFreeHost(ptr));
        }
    }

    void make_pipe() {
        for (auto& bindings : input_bindings) {
            void* d_ptr;
            cuda_check(cudaMalloc(&d_ptr, bindings.size * bindings.dsize));
            device_ptrs.push_back(d_ptr);
        }

        for (auto& bindings : output_bindings) {
            void* d_ptr, * h_ptr;
            size_t size = bindings.size * bindings.dsize;
            cuda_check(cudaMalloc(&d_ptr, size));
            cuda_check(cudaHostAlloc(&h_ptr, size, 0));
            device_ptrs.push_back(d_ptr);
            host_ptrs.push_back(h_ptr);
        }
    }

    void load_from_mat(const Mat& image, Size& size) {
        Mat nchw;
        letterbox(image, nchw, size);
        context->setBindingDimensions(0, nvinfer1::Dims{ 4, {1, 3, size.height, size.width} });
        cuda_check(cudaMemcpyAsync(
            device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, stream));
    }

    void letterbox(const Mat& image, Mat& out, Size& size) {
        float inp_h = size.height, inp_w = size.width;
        float height = image.rows, width = image.cols;

        float r = min(inp_h / height, inp_w / width);
        int padw = round(width * r), padh = round(height * r);

        Mat tmp;
        if (static_cast<int>(width) != padw || static_cast<int>(height) != padh) {
            resize(image, tmp, Size(padw, padh));
        }
        else  tmp = image.clone();

        float dw = inp_w - padw, dh = inp_h - padh;
        dw /= 2.0f, dh /= 2.0f;

        int top = round(dh);
        int bottom = round(dh);
        int left = round(dw);
        int right = round(dw);

        copyMakeBorder(tmp, tmp, top, bottom, left, right, BORDER_CONSTANT, { 114, 114, 114 });

        dnn::blobFromImage(tmp, out, 1 / 255.f, Size(), Scalar(0, 0, 0), true, false, CV_32F);
        pparam.ratio = 1 / r;
        pparam.dw = dw;
        pparam.dh = dh;
        pparam.height = height;
        pparam.width = width;
    }


    void infer() {
        context->enqueueV2(device_ptrs.data(), stream, nullptr);
        for (int i = 0; i < num_outputs; i++) {
            size_t osize = output_bindings[i].size * output_bindings[i].dsize;
            cuda_check(cudaMemcpyAsync(
                host_ptrs[i], device_ptrs[i + num_inputs], osize, cudaMemcpyDeviceToHost, stream));
        }
        cudaStreamSynchronize(stream);
    }

    void postprocess(vector<Object>& objs, float score_thres = 0.4f, float iou_thres = 0.6f, int topk = 100) {
        objs.clear();
        auto num_channels = output_bindings[0].dims.d[1];
        auto num_anchors = output_bindings[0].dims.d[2];

        float& dw = pparam.dw, & dh = pparam.dh;
        float& width = pparam.width, & height = pparam.height;
        float& ratio = pparam.ratio;

        vector<Rect> bboxes;
        vector<float> scores;
        vector<int> labels;
        vector<int> indices;
        vector<vector<float>> kpss;

        Mat output = Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(host_ptrs[0]));
        output = output.t();
        for (int i = 0; i < num_anchors; i++) {
            auto row_ptr = output.row(i).ptr<float>();
            auto bboxes_ptr = row_ptr;
            auto scores_ptr = row_ptr + 4;
            auto kps_ptr = row_ptr + 5;

            float score = *scores_ptr;
            if (score > score_thres) {
                float x = *bboxes_ptr++ - dw;
                float y = *bboxes_ptr++ - dh;
                float w = *bboxes_ptr++;
                float h = *bboxes_ptr;

                float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
                float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
                float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
                float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

                Rect_<float> bbox;
                bbox.x = x0;
                bbox.y = y0;
                bbox.width = x1 - x0;
                bbox.height = y1 - y0;
                vector<float> kps;
                for (int k = 0; k < 17; k++) {
                    float kps_x = (*(kps_ptr + 3 * k) - dw) * ratio;
                    float kps_y = (*(kps_ptr + 3 * k + 1) - dh) * ratio;
                    float kps_s = *(kps_ptr + 3 * k + 2);
                    kps_x = clamp(kps_x, 0.f, width);
                    kps_y = clamp(kps_y, 0.f, height);
                    kps.push_back(kps_x);
                    kps.push_back(kps_y);
                    kps.push_back(kps_s);
                }

                bboxes.emplace_back(bbox);
                labels.push_back(0);
                scores.push_back(score);
                kpss.emplace_back(kps);
            }
        }
        dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
        int cnt = 0;
        for (auto& i : indices) {
            if (cnt >= topk)  break;
            Object obj;
            obj.rect = bboxes[i];
            obj.prob = scores[i];
            obj.label = labels[i];
            obj.kps = kpss[i];
            objs.emplace_back(obj);
            cnt += 1;
        }
    }

    void draw_objects(Mat& res, const vector<Object>& objs) {
        const int num_point = 17;
        for (auto& obj : objs) {
            rectangle(res, obj.rect, { 0, 0, 255 }, 2);
            string text = "person " + to_string(obj.prob * 100) + "%";
            int baseLine = 0;
            Size label_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

            int x = static_cast<int>(obj.rect.x);
            int y = static_cast<int>(obj.rect.y);


            rectangle(res, Rect(x, y, label_size.width, label_size.height + baseLine), { 0, 0, 255 }, -1);
            putText(res, text, Point(x, y + label_size.height), FONT_HERSHEY_SIMPLEX, 0.4, { 255, 255, 255 }, 1);

            auto& kps = obj.kps;
            for (int k = 0; k < num_point + 2; k++) {
                if (k < num_point) {
                    int   kps_x = round(kps[k * 3]);
                    int   kps_y = round(kps[k * 3 + 1]);
                    float kps_s = kps[k * 3 + 2];
                    if (kps_s > 0.5f) {
                        circle(res, { kps_x, kps_y }, 5, KPS_COLORS[k], -1);
                    }
                }
                auto& ske = SKELETON[k];
                int pos1_x = round(kps[(ske[0] - 1) * 3]);
                int pos1_y = round(kps[(ske[0] - 1) * 3 + 1]);

                int pos2_x = round(kps[(ske[1] - 1) * 3]);
                int pos2_y = round(kps[(ske[1] - 1) * 3 + 1]);

                float pos1_s = kps[(ske[0] - 1) * 3 + 2];
                float pos2_s = kps[(ske[1] - 1) * 3 + 2];

                if (pos1_s > 0.5f && pos2_s > 0.5f) {
                    line(res, { pos1_x, pos1_y }, { pos2_x, pos2_y }, LIMB_COLORS[k], 2);
                }
            }
        }
    }
};