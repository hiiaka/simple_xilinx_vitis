#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <chrono>

#include "common.h"

const std::string baseImagePath = "./images/";

/**
 * @brief 指定したディレクトリの画像ファイル名の一覧を取得する
 *
 * @param path - 入力画像が格納されたディレクトリへのパス
 * @param images - 画像ファイル名をベクトル形式で返す
 *
 * @return 
 */
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

/**
 * @brief Softmax関数
 *
 * @param data - Softmax関数への入力値
 * @param size - 要素数
 * @param result - Softmax関数の出力
 *
 * @return
 */
void CPUCalcSoftmax(const float* data, size_t size, float* result) {
  assert(data && result);
  double sum = 0.0f;

  for (size_t i = 0; i < size; i++) {
    result[i] = exp(data[i]);
    sum += result[i];
  }
  for (size_t i = 0; i < size; i++) {
    result[i] /= sum;
  }
}

/**
 * @brief 配列の最大値のインデックスを求める
 *
 * @param data - pointer 配列
 * @param size - 配列の要素数
 *
 * @return 配列の最大値のインデックス
 */
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

/**
 * @brief MNISTの推論処理を実行
 *
 * @param runner - Runnerのポインタ
 *
 * @return
 */
void runMNIST(vart::Runner* runner) {
  std::vector<string> images;

  ListImages(baseImagePath, images);
  if (images.size() == 0) {
    std::cerr << "\nError: No images existing under " << baseImagePath << endl;
    return;
  }

  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_dims();  // {1, 1, 1, 10}
  auto in_dims = inputTensors[0]->get_dims();    // {1, 28, 28, 1}
  int batchSize = in_dims[0];
  int outSize = out_dims[0] * out_dims[1] * out_dims[2] * out_dims[3];    // 10
  int inSize = in_dims[0] * in_dims[1] * in_dims[2] * in_dims[3];      // 784
  int inHeight = in_dims[1];  // 28
  int inWidth = in_dims[2];    // 28

  printf("out_dims : %d, %d, %d, %d\n", out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
  printf("in_dims  : %d, %d, %d, %d\n", in_dims[0], in_dims[1], in_dims[2], in_dims[3]);

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;

  float* softmax = new float[outSize];
  float* imageInputs = new float[inSize];
  float* FCResult = new float[outSize];
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;
  batchTensors.push_back(std::shared_ptr<xir::Tensor>(
    xir::Tensor::create(inputTensors[0]->get_name(), in_dims, 
                        xir::DataType::FLOAT, sizeof(float) * 8u)));
  batchTensors.push_back(std::shared_ptr<xir::Tensor>(
    xir::Tensor::create(outputTensors[0]->get_name(), out_dims, 
                        xir::DataType::FLOAT, sizeof(float) * 8u)));

  int count = 0;
  double elapsed = 0;
  for (int i = 0; i < images.size(); i++) {
    cv::Mat input_image = cv::imread(baseImagePath + images[i], 0);
    input_image.convertTo(input_image, CV_32F, 1.0 / 255);
    memcpy(imageInputs, input_image.data, 
      sizeof(float) * inHeight * inWidth);

    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
      imageInputs, batchTensors[0].get()));                      
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
      FCResult, batchTensors[1].get()));
    inputsPtr.clear();
    outputsPtr.clear();
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    std::chrono::system_clock::time_point  start, end;

    start = std::chrono::system_clock::now();
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);
    end = std::chrono::system_clock::now();
    elapsed += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

    std::cout << "\nImage : " << images[i] << ", label = ";
    CPUCalcSoftmax(FCResult, outSize, softmax);
    int max = argmax(softmax, outSize);
    std::cout << max << std::endl;
    count++;
    
    inputs.clear();
    outputs.clear();

  }
  std::cout << "Elapsed Time per frame: " << elapsed / count << "[us]" << std::endl;
  delete[] FCResult;
  delete[] imageInputs;
  delete[] softmax;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage of MNIST demo: ./MNIST [model_file]" << endl;
    return -1;
  }
  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
    << "MNIST should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();
  
  auto runner = vart::Runner::create_runner(subgraph[0], "run");

  runMNIST(runner.get());
  return 0;
}
