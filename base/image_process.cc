#include "image_process.h"
#include "iostream"
#include <limits>
#include "util.h"

// temp: for rebuttal
#include <fstream>
using std::ofstream;

using std::numeric_limits;
using std::endl;
using std::cerr;
using cv::Mat;
using Eigen::Vector2i;
using Eigen::Vector2d;
using Eigen::Vector3d;

namespace {
} // empty namespace

namespace DPM {

void VisualizeDepthImage(const Mat& img_depth, Mat& img_colored) {
  Mat img_normalized;
  img_normalized.create(img_depth.rows, img_depth.cols, CV_8UC1);
  img_normalized.setTo(cv::Scalar(0));
  float minvalue = numeric_limits<float>::max();
  float maxvalue = -numeric_limits<float>::max();
  for (int i = 0; i < img_depth.rows; ++i)
    for (int j = 0; j < img_depth.cols; ++j) {
      if (img_depth.at<float>(i, j) == 0)
        continue;
      minvalue = std::min(minvalue, img_depth.at<float>(i, j));
      maxvalue = std::max(maxvalue, img_depth.at<float>(i, j));
    }
  for (int i = 0; i < img_depth.rows; ++i)
    for (int j = 0; j < img_depth.cols; ++j) {
      if (img_depth.at<float>(i, j) == 0) {
        img_normalized.at<uchar>(i, j) = 0;
        continue;
      }
      img_normalized.at<uchar>(i, j) = static_cast<int>(255 * ((img_depth.at<float>(i, j) - minvalue) / (maxvalue - minvalue)));
    }
  img_colored.create(img_normalized.rows, img_normalized.cols, CV_8UC3);
  img_colored.setTo(cv::Vec3b(0, 0, 0));
  for (int y = 0; y < img_normalized.rows; ++y)
    for (int x = 0; x < img_normalized.cols; ++x) {
      const int gray = img_normalized.at<uchar>(y, x);
      int red, green, blue;
      if (gray == 0) {
        red = 255; green = 255; blue = 255;
      }
      else if (gray < 128) {
        red = 0; green = 2 * gray; blue = 255 - green;
      }
      else {
        blue = 0; red = (gray - 128) * 2; green = 255 - red;
      }
      img_colored.at<cv::Vec3b>(y, x) = cv::Vec3b(blue, green, red);
    }
}

bool Recenteralize(const float k_margin, cv::Mat& img, Vector2i& offset) {
  cv::Mat img_cp = img.clone();
  int width = img_cp.cols;
  int height = img_cp.rows;
  offset[0] = 0;
  offset[1] = 0;

  Vector2i bb_tl(numeric_limits<int>::max(), numeric_limits<int>::max());
  Vector2i bb_br(-numeric_limits<int>::max(), -numeric_limits<int>::max());
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      if (img_cp.at<float>(y, x) > 0 ) {
        bb_tl[0] = std::min(bb_tl[0], y);
        bb_tl[1] = std::min(bb_tl[1], x);
        bb_br[0] = std::max(bb_br[0], y);
        bb_br[1] = std::max(bb_br[1], x);
      }

  if (bb_tl[0] >= 0 && bb_tl[1] >= 0 && bb_br[0] <= (height - 1) && bb_br[1] <= (width - 1) &&
      bb_br[0] - bb_tl[0] > 2 && bb_br[1] - bb_tl[1] > 2) {
  // if (bb_tl[0] > 1 && bb_tl[1] > 1 && bb_br[0] < (height - 1) && bb_br[1] < (width - 1) &&
  //     bb_br[0] - bb_tl[0] > 2 && bb_br[1] - bb_tl[1] > 2) {
    cv::Mat img_new;
    int h_range = bb_br[0] - bb_tl[0];
    int w_range = bb_br[1] - bb_tl[1];
    int h_new = std::max(h_range + 1, w_range + 1) * (1 + k_margin);
    int w_new = h_new;
    int h_start = (h_new - h_range) / 2;
    int w_start = (w_new - w_range) / 2;
    img_new.create(h_new, w_new, img.type());
    img_new.setTo(0);
    for (int y = 0; y <= h_range; ++y)
      for (int x = 0; x <= w_range; ++x)
        img_new.at<float>(y + h_start, x + w_start) = img.at<float>(bb_tl[0] + y, bb_tl[1] + x);
    //Note: offset order: first x, then y
    offset[0] = bb_tl[1] - w_start;
    offset[1] = bb_tl[0] - h_start;
    img = img_new;
  }
  return true;
}

bool ComputeNormalImg(const cv::Mat& img_depth, const double kPixelsBetwPts, cv::Mat& img_normal)
{
  cv::Mat mask(img_depth.rows, img_depth.cols, CV_8UC1);
  mask.setTo(0);
  mask.setTo(255, img_depth > 0);
  const double rStepSize = 0.5;
  double step = rStepSize / kPixelsBetwPts;
  img_normal.create(img_depth.rows, img_depth.cols, CV_8UC3);
  img_normal.setTo(cv::Vec3b(255, 255, 255));
  std::vector<Vector2i> neighbor(8);
  neighbor[0] = Vector2i(-1, -1); neighbor[1] = Vector2i(-1, 0); neighbor[2] = Vector2i(-1, 1); neighbor[3] = Vector2i(0, 1);
  neighbor[4] = Vector2i(1, 1); neighbor[5] = Vector2i(1, 0); neighbor[6] = Vector2i(1, -1); neighbor[7] = Vector2i(0, -1);
  #pragma omp parallel for
  for (int i = 1; i < img_depth.rows - 1; ++i) {
    for (int j = 1; j < img_depth.cols - 1; ++j) {
      std::vector<Vector3d> normals_neighbor;
      double h_c = img_depth.at<float>(i, j);
      for (int ind = 0; ind < 8; ++ind) {
        int ind_first = ind % 8;
        int ind_second = (ind + 1) % 8;
        double h_1 = img_depth.at<float>(i + neighbor[ind_first][0], j + neighbor[ind_first][1]);
        double h_2 = img_depth.at<float>(i + neighbor[ind_second][0], j + neighbor[ind_second][1]);
        const Vector3d diff1 = Vector3d(neighbor[ind_first][0] * step, neighbor[ind_first][1] * step, h_1 - h_c);
        const Vector3d diff2 = Vector3d(neighbor[ind_second][0] * step, neighbor[ind_second][1] * step, h_2 - h_c);
        Vector3d normal = -diff1.cross(diff2);
        if (normal[2] < 0.0)
          normal = -normal;
        const double norm = normal.norm();
        if (norm == 0.0)
          continue;
        normal /= norm;
        normals_neighbor.push_back(normal);
      }
      Vector3d normal_mean(0, 0, 0);
      for (auto normal : normals_neighbor)
        normal_mean += normal;
      normal_mean = normal_mean / normals_neighbor.size();
      const double norm = normal_mean.norm();
      if (norm == 0.0)
        continue;
      normal_mean /= norm;
      cv::Vec3b color;
      color[0] = static_cast<uchar>(round((normal_mean[0] + 1.0) / 2.0 * 255.0));
      color[1] = static_cast<uchar>(round((normal_mean[1] + 1.0) / 2.0 * 255.0));
      color[2] = static_cast<uchar>(round(normal_mean[2] * 255.0));
      img_normal.at<cv::Vec3b>(i, j) = color;
    }
  }
  img_normal.setTo(cv::Vec3b(255, 255, 255), mask == 0);
  return true;
}

bool ComputeGradient(const cv::Mat& img_depth, cv::Mat& img_gradient) {
  Mat img_normalized;
  img_normalized.create(img_depth.rows, img_depth.cols, CV_8UC1);
  img_normalized.setTo(cv::Scalar(0));
  float minvalue = numeric_limits<float>::max();
  float maxvalue = -numeric_limits<float>::max();
  for (int i = 0; i < img_depth.rows; ++i)
    for (int j = 0; j < img_depth.cols; ++j) {
      if (img_depth.at<float>(i, j) == 0)
        continue;
      minvalue = std::min(minvalue, img_depth.at<float>(i, j));
      maxvalue = std::max(maxvalue, img_depth.at<float>(i, j));
    }
  minvalue = 0.0;
  for (int i = 0; i < img_depth.rows; ++i)
    for (int j = 0; j < img_depth.cols; ++j) {
      if (img_depth.at<float>(i, j) == 0) {
        img_normalized.at<uchar>(i, j) = 0;
        continue;
      }
      img_normalized.at<uchar>(i, j) = static_cast<int>(128 * ((img_depth.at<float>(i, j) - minvalue) / (maxvalue - minvalue))) + 127;
    }
  Mat grad;
  grad.create(img_depth.rows, img_depth.cols, CV_8UC1);
  grad.setTo(cv::Scalar(0));
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  // GaussianBlur( img_normalized, img_normalized, Size(3, 3), 0, 0, cv::BORDER_DEFAULT );
  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  /// Gradient X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
  cv::Sobel( img_normalized, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( grad_x, abs_grad_x );
  // std::string ty = type2str(abs_grad_x.type());
  // printf("abs_grad_x matrix: %s %dx%d \n", ty.c_str(), abs_grad_x.cols, abs_grad_x.rows );

  float sum3 = 0.0;
  for (int y = 0; y < abs_grad_x.rows; ++y)
    for (int x = 0; x < abs_grad_x.cols; ++x) {
      sum3 += abs_grad_x.at<uchar>(y, x);
    }
  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
  cv::Sobel( img_normalized, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( grad_y, abs_grad_y );
  /// Total Gradient (approximate)
  // cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
  for (int j = 0; j < grad.rows; ++j)
    for (int i = 0; i < grad.cols; ++i)
      grad.at<uchar>(j, i) = std::sqrt(abs_grad_x.at<uchar>(j, i) * abs_grad_x.at<uchar>(j, i) +
                                  abs_grad_y.at<uchar>(j, i) * abs_grad_y.at<uchar>(j, i));

  img_gradient = grad;
  return true;
}

double ComputeFootprintDistance(std::vector<Eigen::Vector2d> &footprint_i_first, std::vector<Eigen::Vector2d> &footprint_i_second) {
  double d = numeric_limits<double>::max();
  for(unsigned i = 0; i < 4; ++i)
    for(unsigned j = 0; j < 4; ++j)
      d = std::min((footprint_i_first[i] - footprint_i_second[j]).norm(), d);
  return d;
}

bool SetMiddleIndexFirst(std::vector<Eigen::Vector2d> &footprint_i_first, std::vector<Eigen::Vector2d> &footprint_i_second, std::vector<Eigen::Vector2d> &footprint_i_third) {
  // auto footprint_i_first_cp = footprint_i_first; auto footprint_i_second_cp = footprint_i_second; auto footprint_i_third_cp = footprint_i_third;
  Vector2d bb_tl1(numeric_limits<double>::max(), numeric_limits<double>::max()); Vector2d bb_br1(-numeric_limits<double>::max(), -numeric_limits<double>::max());
  Vector2d bb_tl2(numeric_limits<double>::max(), numeric_limits<double>::max()); Vector2d bb_br2(-numeric_limits<double>::max(), -numeric_limits<double>::max());
  Vector2d bb_tl3(numeric_limits<double>::max(), numeric_limits<double>::max()); Vector2d bb_br3(-numeric_limits<double>::max(), -numeric_limits<double>::max());
  for(auto&& ele : footprint_i_first) {
    bb_tl1[0] = std::min(bb_tl1[0], ele[0]); bb_tl1[1] = std::min(bb_tl1[1], ele[1]);
    bb_br1[0] = std::max(bb_br1[0], ele[0]); bb_br1[1] = std::max(bb_br1[1], ele[1]);    
  }
  for(auto&& ele : footprint_i_second) {
    bb_tl2[0] = std::min(bb_tl2[0], ele[0]); bb_tl2[1] = std::min(bb_tl2[1], ele[1]);
    bb_br2[0] = std::max(bb_br2[0], ele[0]); bb_br2[1] = std::max(bb_br2[1], ele[1]);    
  }
  for(auto&& ele : footprint_i_second) {
    bb_tl3[0] = std::min(bb_tl3[0], ele[0]); bb_tl3[1] = std::min(bb_tl3[1], ele[1]);
    bb_br3[0] = std::max(bb_br3[0], ele[0]); bb_br3[1] = std::max(bb_br3[1], ele[1]);    
  }
  double d12 = 0, d13 = 0, d23 = 0; 
  double d1 = 0, d2 = 0, d3 = 0;
  if (std::min(bb_br1[0], bb_br2[0]) >= std::max(bb_tl1[0], bb_tl2[0]) && std::min(bb_br1[1], bb_br2[1]) >= std::max(bb_tl1[1], bb_tl2[1]))
    d12 = 0;
  else 
    d12 = ComputeFootprintDistance(footprint_i_first, footprint_i_second);

  if (std::min(bb_br1[0], bb_br3[0]) >= std::max(bb_tl1[0], bb_tl3[0]) && std::min(bb_br1[1], bb_br3[1]) >= std::max(bb_tl1[1], bb_tl3[1]))
    d13 = 0;
  else 
    d13 = ComputeFootprintDistance(footprint_i_first, footprint_i_third);

  if (std::min(bb_br3[0], bb_br2[0]) >= std::max(bb_tl3[0], bb_tl2[0]) && std::min(bb_br3[1], bb_br2[1]) >= std::max(bb_tl3[1], bb_tl2[1]))
    d23 = 0;
  else 
    d23 = ComputeFootprintDistance(footprint_i_third, footprint_i_second);
  d1 = d12 + d13; d2 = d12 + d23; d3 = d13 + d23;

  if (d2 < d3 && d2 < d1)
    std::swap(footprint_i_second, footprint_i_first); 
  if (d3 < d1 && d3 < d2)
    std::swap(footprint_i_third, footprint_i_first); 
}


// @usage:
//    string ty = type2str(img_gray.type());
//    printf("img_gray matrix: %s %dx%d \n", ty.c_str(), img_gray.cols, img_gray.rows );
std::string type2str(int type) {
  uchar chans;
  std::string r;
  uchar depth = type & CV_MAT_DEPTH_MASK;
  chans = 1 + (type >> CV_CN_SHIFT);
  switch ( depth ) {
  case CV_8U:  r = "8U"; break;
  case CV_8S:  r = "8S"; break;
  case CV_16U: r = "16U"; break;
  case CV_16S: r = "16S"; break;
  case CV_32S: r = "32S"; break;
  case CV_32F: r = "32F"; break;
  case CV_64F: r = "64F"; break;
  default:     r = "User"; break;
  }
  r += "C";
  r += (chans + '0');
  return r;
}

bool ComputeXYZN(const cv::Mat& img_depth, const double kPixelsBetwPts, const std::string& path_xyzn)
{
  cv::Mat mask(img_depth.rows, img_depth.cols, CV_8UC1);
  cv::Mat img_normal;
  mask.setTo(0);
  mask.setTo(255, img_depth > 0);
  const double rStepSize = 0.5;
  double step = rStepSize / kPixelsBetwPts;
  img_normal.create(img_depth.rows, img_depth.cols, CV_64FC3);
  img_normal.setTo(cv::Vec3d(0.0, 0.0, 1.0));
  std::vector<Vector2i> neighbor(8);
  neighbor[0] = Vector2i(-1, -1); neighbor[1] = Vector2i(-1, 0); neighbor[2] = Vector2i(-1, 1); neighbor[3] = Vector2i(0, 1);
  neighbor[4] = Vector2i(1, 1); neighbor[5] = Vector2i(1, 0); neighbor[6] = Vector2i(1, -1); neighbor[7] = Vector2i(0, -1);
  #pragma omp parallel for
  for (int i = 1; i < img_depth.rows - 1; ++i) {
    for (int j = 1; j < img_depth.cols - 1; ++j) {
      std::vector<Vector3d> normals_neighbor;
      double h_c = img_depth.at<float>(i, j);
      for (int ind = 0; ind < 8; ++ind) {
        int ind_first = ind % 8;
        int ind_second = (ind + 1) % 8;
        double h_1 = img_depth.at<float>(i + neighbor[ind_first][0], j + neighbor[ind_first][1]);
        double h_2 = img_depth.at<float>(i + neighbor[ind_second][0], j + neighbor[ind_second][1]);
        const Vector3d diff1 = Vector3d(neighbor[ind_first][1] * step, neighbor[ind_first][0] * step, h_1 - h_c);
        const Vector3d diff2 = Vector3d(neighbor[ind_second][1] * step, neighbor[ind_second][0] * step, h_2 - h_c);
        Vector3d normal = -diff1.cross(diff2);
        if (normal[2] < 0.0)
          normal = -normal;
        const double norm = normal.norm();
        if (norm == 0.0)
          continue;
        normal /= norm;
        normals_neighbor.push_back(normal);
      }
      Vector3d normal_mean(0, 0, 0);
      for (auto normal : normals_neighbor)
        normal_mean += normal;
      normal_mean = normal_mean / normals_neighbor.size();
      const double norm = normal_mean.norm();
      if (norm == 0.0)
        continue;
      normal_mean /= norm;
      img_normal.at<cv::Vec3d>(i, j)[0] = normal_mean[0];
      img_normal.at<cv::Vec3d>(i, j)[1] = normal_mean[1];
      img_normal.at<cv::Vec3d>(i, j)[2] = normal_mean[2];
    }
  }
  img_normal.setTo(cv::Vec3d(0.0, 0.0, 1.0), mask == 0);
  std::ofstream ofstr(path_xyzn);
  if (!ofstr.is_open())
    return false;
  ofstr << "# ground 0.0" << std::endl;
  for (int i = 0; i < img_depth.rows; ++i) {
    for (int j = 0; j < img_depth.cols; ++j) {
      ofstr << j << " " << i << " " << img_depth.at<float>(i, j) << " " << 
        img_normal.at<cv::Vec3d>(i, j)[0] << " " <<
        img_normal.at<cv::Vec3d>(i, j)[1] << " " <<
        img_normal.at<cv::Vec3d>(i, j)[2] << " " << endl;
    }
  }
  ofstr.close();

  return true;
}


} // DPM

bool ComputeNormalImgF(float *image_depth, double kPixelsBetwPts, int width, int height, unsigned char *image_normal) {
  // float pointer to 
  cv::Mat img_depth;
  img_depth.create(height, width, CV_32F);
  img_depth.setTo(0);
  for (int i = 0; i < height; ++i)
    for (int j = 0; j < width; ++j)
      img_depth.at<float>(i, j) = image_depth[i*width+j];

  cv::Mat img_normal;
  if(!DPM::ComputeNormalImg(img_depth, kPixelsBetwPts, img_normal))
    return false;
  // image_normal = new unsigned char[img_normal.rows * img_normal.cols * 3];
  // image_normal = (unsigned char*) malloc (img_normal.rows * img_normal.cols * 3);
  for (int i = 0; i < img_normal.rows; ++i)
    for (int j = 0; j < img_normal.cols; ++j){
      image_normal[i*img_normal.cols*3+j*3+0]=(img_normal.at<cv::Vec3b>(i, j)[0]);
      image_normal[i*img_normal.cols*3+j*3+1]=(img_normal.at<cv::Vec3b>(i, j)[1]);
      image_normal[i*img_normal.cols*3+j*3+2]=(img_normal.at<cv::Vec3b>(i, j)[2]);
    }
  return true;
}

extern "C"
{
    extern bool cffi_ComputeNormalImgF(float *image_depth, double kPixelsBetwPts, int width, int height, unsigned char *image_normal) {
        return ComputeNormalImgF(image_depth, kPixelsBetwPts, width, height, image_normal);
    }
}
