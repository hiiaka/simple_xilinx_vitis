#include "utils.hpp"

#include <chrono>
#include <assert.h>
#include <iostream>

inline float sigmoid(float x) {
  return 1.0 / (1 + exp(-x));
}

int argmax(const float* data, int size) {
  float max = 0;
  int result = 0;
  for (int j = 0; j < size; j++) {
    if (data[j] > max) {
      max = data[j];
      result = j;
      }
  }
  return result;
}

inline float overlap(float x1, float w1, float x2, float w2) {
  float left = max(x1 - w1 / 2.0, x2 - w2 / 2.0);
  float right = min(x1 + w1 / 2.0, x2 + w2 / 2.0);
  return right - left;
}

inline float cal_iou(const std::vector<float>& box1, const std::vector<float>& box2) {
  float w = overlap(box1[0], box1[2], box2[0], box2[2]);
  float h = overlap(box1[1], box1[3], box2[1], box2[3]);
  if (w < 0 || h < 0) {
    return 0;
  }

  float inter_area = w * h;
  float union_area = box1[2] * box1[3] + box2[2] * box2[3] - inter_area;
  return inter_area * 1.0 / union_area;
}

void get_anchors(const std::string& file_path, std::vector<float>& anchors) {
  std::fstream ifs(file_path);
  assert(ifs);
    
  std::string str;
  std::getline(ifs, str);

  std::string tmp;
  std::istringstream stream(str);
  while (std::getline(stream, tmp, ',')) {
    anchors.push_back(std::stof(tmp));
  }
}

void get_classes(const std::string& file_path, std::vector<std::string>& classes) {
  std::fstream ifs(file_path);
  assert(ifs);

  std::string str;
  while (std::getline(ifs, str)) {
    classes.push_back(str);
  }
}

void ListImages (const std::string& path, std::vector<std::string>& images) {
  images.clear();
  struct dirent* entry;

  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR* dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      std::string name = entry->d_name;
      std::string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") ||
        (ext == "bmp") || (ext == "PNG") || (ext == "png")) {
          images.push_back(name);
      }
    }
  }
}

void preprocess_image(cv::Mat& raw_image, image& processed_image) {
  int width = raw_image.cols;
  int height = raw_image.rows;
  processed_image.w = width;
  processed_image.h = height;
  processed_image.c = CHANNEL;

  float scale_w = WIDTH / (float)width;
  float scale_h = HEIGHT / (float)height;
  float scale = std::min(scale_w, scale_h);

  cv::Mat resized_image;
  cv::resize(raw_image, resized_image, cv::Size(), scale, scale);  // 幅もしくは高さを416にリサイズ

  int dx = (WIDTH - resized_image.cols + 1) / 2;
  int dy = (HEIGHT - resized_image.rows + 1) / 2;

  // 416x416画像に埋め込み
  for (int h = 0; h < HEIGHT; h++) {
    for (int w = 0; w < WIDTH; w++) {
      for (int c = 0; c < CHANNEL; c++) {
        if ((w >= dx && w < (WIDTH - dx)) && (h >= dy && h < (HEIGHT - dy))) {
          processed_image.data[h * WIDTH * CHANNEL + w * CHANNEL + c] = resized_image.at<cv::Vec3b>(h - dy, w - dx)[CHANNEL - c - 1] / 255.0;  // BGR -> RGB
        } else {
          processed_image.data[h * WIDTH * CHANNEL + w * CHANNEL + c] = 128 / 255.0;
        }
      }
    }
  }
}

void extract_boxes(vart::Runner* runner, std::vector<float*> tensors, std::vector<std::vector<float>>& boxes, std::vector<float> anchors) {
  auto outputTensors = runner->get_output_tensors();
  std::vector<int> out_dims[3] = {
    outputTensors[0]->get_dims(),
    outputTensors[1]->get_dims(),
    outputTensors[2]->get_dims()
  };

  for (int i = 0; i < 3; i++) {  // 13 -> 26 -> 52
    int width = out_dims[i][2];
    int height = out_dims[i][1];
    int channel = out_dims[i][3];
    int sizeOut = channel * width * height;

    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int b = 0; b < 3; b++) {
          float obj_score = sigmoid(tensors[i][h * width * channel + w * channel + b * (channel / 3) + 4]);
          if (obj_score < CONF) {
            continue;
          }

          std::vector<float> box;
          box.push_back((w + sigmoid(tensors[i][h * width * channel + w * channel + b * (channel / 3)])) / width);
          box.push_back((h + sigmoid(tensors[i][h * width * channel + w * channel + b * (channel / 3) + 1])) / height);
          box.push_back(anchors[(12 - 6 * i) + b * 2] * exp(tensors[i][h * width * channel + w * channel + b * (channel / 3) + 2]) / WIDTH);
          box.push_back(anchors[(12 - 6 * i) + b * 2 + 1] * exp(tensors[i][h * width * channel + w * channel + b * (channel / 3) + 3]) / HEIGHT);
          box.push_back(obj_score);
          box.push_back(0);
          for (int k = 0; k < channel / 3 - 5; k++) {
            box.push_back(obj_score * sigmoid(tensors[i][h * width * channel + w * channel + b * (channel / 3) + 5 + k]));
          }

          boxes.push_back(box);

        }
      }
    }
  }
}

void transform_boxes(std::vector<std::vector<float>>& boxes, int sWidth, int sHeight) {
  float scale_w = WIDTH / (float)sWidth;
  float scale_h = HEIGHT / (float)sHeight;
  float scale = std::min(scale_w, scale_h);
  int dx = (WIDTH - sWidth * scale) / 2;
  int dy = (HEIGHT - sHeight * scale) / 2;
  for (std::vector<float>& box : boxes) {
    box[0] = (box[0] * WIDTH - dx) / scale;  // x座標
    box[1] = (box[1] * HEIGHT - dy) / scale;  // y座標
    box[2] *= sWidth;  // 幅
    box[3] *= sHeight;  // 高さ
  }
}

std::vector<std::vector<float>> applyNMS(std::vector<std::vector<float>>& boxes, int class_num, const float thresh) {
  std::vector<std::pair<int, float>> order(boxes.size());
  std::vector<std::vector<float>> result;

  for (int k = 0; k < class_num; k++) {
    for (int i = 0; i < boxes.size(); i++) {
      order[i].first = i;
      order[i].second = boxes[i][k + 6];
    }

    // orderをクラス確率の降順でソート
    std::sort(order.begin(), order.end(), 
      [](const std::pair<int, float> &ls, const std::pair<int, float> &rs) {return ls.second > rs.second;});

    std::vector<bool> exist_box(boxes.size(), true);

    for (int i = 0; i < boxes.size(); i++) {
      int _i = order[i].first;
      if (!exist_box[_i]) {
        continue;
      }
      if (boxes[_i][6 + k] < CONF) {
        exist_box[_i] = false;
        continue;
      }
      boxes[_i][5] = k;
      result.push_back(boxes[_i]);
      for (int j = i + 1; j < boxes.size(); j++) {
        int _j = order[j].first;
        if (!exist_box[_j]) {
          continue;
        }
        float ovr = cal_iou(boxes[_i], boxes[_j]);
        if (ovr >= thresh) {
          exist_box[_j] = false;
        }
      }
    }
  }

  return result;

}

cv::Scalar hsv_to_bgr(int h, int s, int v) {
  cv::Mat hsv_mat(1, 1, CV_8UC3);
  cv::Mat bgr_mat(1, 1, CV_8UC3);
  hsv_mat.data[0] = h;
  hsv_mat.data[1] = s;
  hsv_mat.data[2] = v;
  cv::cvtColor(hsv_mat, bgr_mat, CV_HSV2BGR);

  return cv::Scalar(bgr_mat.data[0], bgr_mat.data[1], bgr_mat.data[2]);
} 

void draw_box(cv::Mat& image, std::vector<std::vector<float>>& boxes, std::vector<std::string>& classes, int sWidth, int sHeight) {
  for (std::vector<float>& box : boxes) {
    int xmin = box[0] - box[2] / 2.0;
    int ymin = box[1] - box[3] / 2.0;
    int xmax = box[0] + box[2] / 2.0;
    int ymax = box[1] + box[3] / 2.0;

    float* box_ptr = box.data();

    int class_index = box[5];

    cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), hsv_to_bgr((180 * class_index) / 80, 255, 255), 2, 1, 0);
    cv::rectangle(image, cv::Point(xmin, ymin - 15), cv::Point(xmin + 100, ymin), hsv_to_bgr((180 * class_index) / 80, 255, 255), CV_FILLED, 1, 0);
    cv::putText(image, classes[class_index].c_str(), cv::Point(xmin, ymin), 1, 1, cv::Scalar(0, 0, 0), 1);
  }
}

void analyzeOutput(vart::Runner* runner, cv::Mat& image, std::vector<float*> tensors, int sWidth, int sHeight, std::vector<float> anchors, std::vector<std::string>& classes) {
  std::vector<std::vector<float>> boxes;
  extract_boxes(runner, tensors, boxes, anchors);
  transform_boxes(boxes, sWidth, sHeight);
  std::vector<std::vector<float>> selected_boxes = applyNMS(boxes, 80, 0.3);
  draw_box(image, selected_boxes, classes, sWidth, sHeight);
}

void runYOLO(vart::Runner* runner) {
  // get in/out tensor and dims
  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  std::vector<std::int32_t> out_dims0 = outputTensors[0]->get_dims();
  std::vector<std::int32_t> out_dims1 = outputTensors[1]->get_dims();
  std::vector<std::int32_t> out_dims2 = outputTensors[2]->get_dims();
  std::vector<std::int32_t> in_dims = inputTensors[0]->get_dims();

  printf("in_dims   : %d, %d, %d, %d\n", in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
  printf("out_dims0 : %d, %d, %d, %d\n", out_dims0[0], out_dims0[1], out_dims0[2], out_dims0[3]);
  printf("out_dims1 : %d, %d, %d, %d\n", out_dims1[0], out_dims1[1], out_dims1[2], out_dims1[3]);
  printf("out_dims2 : %d, %d, %d, %d\n", out_dims2[0], out_dims2[1], out_dims2[2], out_dims2[3]);

  float* data = new float[in_dims[0] * in_dims[1] * in_dims[2] * in_dims[3]];
  float* result0 = new float[out_dims0[0] * out_dims0[1] * out_dims0[2] * out_dims0[3]];
  float* result1 = new float[out_dims1[0] * out_dims1[1] * out_dims1[2] * out_dims1[3]];
  float* result2 = new float[out_dims2[0] * out_dims2[1] * out_dims2[2] * out_dims2[3]];

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

  batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(inputTensors[0]->get_name(), in_dims, xir::DataType::FLOAT, sizeof(float) * 8u)));
  batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(outputTensors[0]->get_name(), out_dims0, xir::DataType::FLOAT, sizeof(float) * 8u)));
  batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(outputTensors[1]->get_name(), out_dims1, xir::DataType::FLOAT, sizeof(float) * 8u)));
  batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(outputTensors[2]->get_name(), out_dims2, xir::DataType::FLOAT, sizeof(float) * 8u)));

  std::vector<std::string> image_list;
  std::vector<float> anchors;
  std::vector<std::string> classes;
  get_anchors(anchor_path, anchors);
  get_classes(class_path, classes);
  ListImages(image_path, image_list);

  int count = 0;
  double elapsed = 0;
  for (std::string& image_name : image_list) {
    // 画像の読み込みと前処理
    cv::Mat raw_image = cv::imread(image_path + image_name);
    image processed_image;
    preprocess_image(raw_image, processed_image);
    memcpy(data, processed_image.data, sizeof(float) * in_dims[0] * in_dims[1] * in_dims[2] * in_dims[3]);
    
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(data, batchTensors[0].get()));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(result0, batchTensors[1].get()));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(result1, batchTensors[2].get()));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(result2, batchTensors[3].get()));

    inputsPtr.clear();
    outputsPtr.clear();
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());
    outputsPtr.push_back(outputs[1].get());
    outputsPtr.push_back(outputs[2].get());

    std::chrono::system_clock::time_point  start, end;
    std::cout << "Infering " << image_name << "..." << std::endl;
    start = std::chrono::system_clock::now();
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);  // Run
    runner->wait(job_id.first, -1);
    end = std::chrono::system_clock::now();

    elapsed += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    count++;

    std::vector<float*> result;
    result.push_back(result0);
    result.push_back(result1);
    result.push_back(result2);

    analyzeOutput(runner, raw_image, result, processed_image.w, processed_image.h, anchors, classes);

    cv::imwrite(out_path + image_name, raw_image);

  }
  std::cout << "Elapsed Time per frame: " << elapsed / count << "[us]" << std::endl;

  delete[] data;
  delete[] result0;
  delete[] result1;
  delete[] result2;
}
