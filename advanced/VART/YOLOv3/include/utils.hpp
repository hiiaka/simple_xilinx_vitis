#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include "common.h"

#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>
#include <string>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>

#define HEIGHT 416
#define WIDTH 416
#define CHANNEL 3
#define CONF 0.4f

typedef struct _image {
  int w;
  int h;
  int c;
  float data[HEIGHT * WIDTH * CHANNEL];
} image;

const std::string image_path = "./images/";
const std::string anchor_path = "./yolo_anchors.txt";
const std::string class_path = "./coco_classes.txt";
const std::string out_path = "./outputs/";

/**
 * @brief Sigmoid関数
 *
 * @param x - 入力変数
 *
 * @return Sigmoid関数の出力
 */
inline float sigmoid(float x);

/**
 * @brief 配列の最大値のインデックスを求める
 *
 * @param data - 配列
 * @param size - 配列の要素数
 *
 * @return 配列の最大値のインデックス
 */
int argmax(const float* data, int size);

/**
 * @brief 2辺で重なってる部分の長さを出力する
 *
 * @param x1 - 辺の中心座標値
 * @param w1 - 辺の長さ
 * @param x2 - 辺の中心座標値
 * @param w2 - 辺の長さ
 *
 * @return 重なっている部分の辺の長さ
 */
inline float overlap(float x1, float w1, float x2, float w2);

/**
 * @brief 2つのboxのIOUを出力する。
 *
 * @param box1 - box1
 * @param box2 - box2
 *
 * @return IOU値
 */
inline float cal_iou(const std::vector<float>& box1, const std::vector<float>& box2);

/**
 * @brief アンカーを取得する
 *
 * @param file_path - アンカーが書かれたテキストファイルへのパス
 * @param anchors - 読み込んだアンカーをベクトル形式で返す
 *
 * @return 
 */
void get_anchors(const std::string& file_path, std::vector<float>& anchors);

/**
 * @brief クラスを取得する
 *
 * @param file_path - クラスが書かれたテキストファイルへのパス
 * @param classes - 読み込んだクラスをベクトル形式で返す
 *
 * @return 
 */
void get_classes(const std::string& file_path, std::vector<std::string>& classes);

/**
 * @brief 指定したディレクトリの画像ファイル名の一覧を取得する
 *
 * @param path - 入力画像が格納されたディレクトリへのパス
 * @param images - 画像ファイル名をベクトル形式で返す
 *
 * @return 
 */
void ListImages (const std::string& path, std::vector<std::string>& images);

/**
 * @brief 画像をYOLOに入力する為の前処理（正規化と416x416への埋め込み）を行う
 *
 * @param raw_image - 読み込んだ画像データ
 * @param processed_image - 前処理後のデータ
 *
 * @return 
 */
void preprocess_image(cv::Mat& raw_image, image& processed_image);

/**
 * @brief 出力されたテンソルから閾値を超えたboxを選出する
 *
 * @param shapes - グラフ情報を格納する構造体
 * @param tensors - 読み込んだ画像データ
 * @param boxes - 前処理後のデータ
 * @param anchors - アンカー
 *
 * @return 
 */
void extract_boxes(vart::Runner* runner, std::vector<float*> tensors, std::vector<std::vector<float>>& boxes, std::vector<float> anchors);

/**
 * @brief boxを入力画像のサイズに合わせてスケーリングと平行移動を行う
 *
 * @param boxes - boxデータ
 * @param sWidth - 入力画像の幅
 * @param sHeight - 入力画像の高さ
 *
 * @return 
 */
void transform_boxes(std::vector<std::vector<float>>& boxes, int sWidth, int sHeight);

/**
 * @brief boxに対してNMSを適用
 *
 * @param boxes - boxデータ
 * @param class_num - クラス数
 * @param thresh - 閾値
 *
 * @return NMS適用後のboxデータ
 */
std::vector<std::vector<float>> applyNMS(std::vector<std::vector<float>>& boxes, int class_num, const float thresh);

/**
 * @brief HSV形式をBGR形式に変換
 *
 * @param h - Hue
 * @param s - Saturation
 * @param v - Value
 *
 * @return BGR形式
 */
cv::Scalar hsv_to_bgr(int h, int s, int v);

/**
 * @brief 入力画像にboxとクラス名の描画を行う
 *
 * @param images - 入力画像データ
 * @param boxes - boxデータ
 * @param classes - クラスデータ
 * @param sWidth - 入力画像の幅
 * @param sHeight - 入力画像の高さ
 *
 * @return
 */
void draw_box(cv::Mat& image, std::vector<std::vector<float>>& boxes, std::vector<std::string>& classes, int sWidth, int sHeight);

/**
 * @brief 出力されたテンソルの後処理を行う
 *
 * @param shapes - グラフ情報を格納する構造体
 * @param images - 入力画像データ
 * @param tensors - 出力のテンソル
 * @param sWidth - 入力画像の幅
 * @param sHeight - 入力画像の高さ
 * @param anchors - アンカーデータ
 * @param classes - クラスデータ
 *
 * @return
 */
void analyzeOutput(vart::Runner* runner, cv::Mat& image, std::vector<float*> tensors, int sWidth, int sHeight, std::vector<float> anchors, std::vector<std::string>& classes);

/**
 * @brief 推論処理をDPUで実行する
 *
 * @param runner - Runnerのポインタ
 *
 * @return
 */
void runYOLO(vart::Runner* runner);

#endif  // __UTILS_HPP__