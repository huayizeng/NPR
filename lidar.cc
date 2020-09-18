#include "lidar.h"
#include <Eigen/Dense>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <queue>
#include <fstream>
#include <iterator>
#include <random>

//TODO(Henry): very hard to debug because of the so-called 'HDF5 library version mismatched error'
//             will modify to #include "H5Cpp.h" if time permits
#include <H5Cpp.h>

#include "image_process.h"

using namespace H5;
using cv::Mat;
using std::cerr;
using std::endl;
using std::make_pair;
using std::pair;
using std::queue;
using std::ifstream;
using std::ofstream;

using Eigen::Vector2i;
using Eigen::Vector3i;
using Eigen::Vector2d;
using Eigen::Vector3d;

namespace {
bool ComparePair(const pair<float, Vector2i>& lhs, const pair<float, Vector2i>& rhs) {return lhs.first < rhs.first;}

bool FindCenterSeed(cv::Mat& img, Vector2i& seed) {
  int height = img.rows;
  int width = img.cols;
  if (img.at<float>(height / 2,  width / 2) > 0)
  {
    seed = Vector2i(height / 2, width / 2);
    return true;
  }

  std::vector<pair<float, Vector2i> > depth_set;
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      if (img.at<float>(y, x) != 0)
        depth_set.push_back(make_pair(img.at<float>(y, x), Vector2i(y, x)));

  if (depth_set.empty())
  {
    cerr << "all depth value are 0" << endl;
    return false;
  }
  float randnum = 0.9;

  int offset = static_cast<int>(round(depth_set.size() * randnum));
  if (offset >= depth_set.size())
    offset = depth_set.size() - 1;
  const std::vector<pair<float, Vector2i> >::iterator position = depth_set.begin() + offset;

  nth_element(depth_set.begin(),
              position,
              depth_set.end(),
              ComparePair);

  double height_offset = (*position).first;
  cerr << "height in seed: " << height_offset << endl;
  seed = (*position).second;
  return true;
}

} // empty namespace

namespace DPM {


bool ReadH5(const std::string& filename, Mat& img) {
  const H5std_string FILE_NAME(filename);
  H5File file( FILE_NAME, H5F_ACC_RDONLY );
  const H5std_string DATASET_NAME_SHAPE( "shape" );
  DataSet dataset_shape = file.openDataSet( DATASET_NAME_SHAPE);
  int shape[2];
  dataset_shape.read( shape, PredType::NATIVE_INT );
  int NX = shape[1];
  int NY = shape[0];
  float data_out[NY][NX]; /* output buffer */
  const H5std_string DATASET_NAME( "matrix" );
  DataSet dataset = file.openDataSet( DATASET_NAME);

  for (int j = 0; j < NY; j++)
    for (int i = 0; i < NX; i++)
      data_out[j][i] = 0;
  dataset.read(data_out, PredType::NATIVE_FLOAT );

  img.create(NY, NX, CV_32F);
  img.setTo(cv::Scalar(0));
  for (int j = 0; j < NY; ++j)
    for (int i = 0; i < NX; ++i)
      img.at<float>(j, i) = data_out[j][i];
  file.close();
  return true;
}

bool WriteH5(const std::string& filename, const cv::Mat& img) {
  H5File file_out(filename.c_str(), H5F_ACC_TRUNC);
  hsize_t dimsf_data[] = {static_cast<hsize_t>(img.cols * img.rows)};
  DataSpace dataspace_data(1, dimsf_data);
  DataType datatype_data(PredType::NATIVE_DOUBLE);
  DataSet dataset_data_out = file_out.createDataSet("matrix", datatype_data, dataspace_data);
  float data_out[img.cols][img.rows];
  for (int j = 0; j < img.cols; ++j)
    for (int i = 0; i < img.rows; ++i)
      data_out[j][i] = img.at<float>(j, i);
  dataset_data_out.write(data_out, PredType::NATIVE_FLOAT);
  dataset_data_out.close();
  dataspace_data.close();

  hsize_t dimsf_shape[] = {2};
  DataSpace dataspace_shape(1, dimsf_shape);
  DataType datatype_out(PredType::NATIVE_INT);
  DataSet dataset_shape_out = file_out.createDataSet("shape", datatype_out, dataspace_shape);
  int shape_out[2] = {img.cols, img.rows};
  dataset_shape_out.write(shape_out, PredType::NATIVE_INT);
  dataset_shape_out.close();
  dataspace_shape.close();
  file_out.close();
  return true;
}


bool Lidar::ReadImgDepth(const std::string& filename_dsm, const std::string& filename_dtm) {
  Mat img_dsm;
  Mat img_dtm;
  if (!ReadH5(filename_dsm, img_dsm)) {
    cerr << "error in reading dsm" << filename_dsm << endl;
    return false;
  }
  if (!ReadH5(filename_dtm, img_dtm)) {
    cerr << "error in reading dtm" << filename_dtm << endl;
    return false;
  }
  img_depth_ = img_dsm - img_dtm;
  return true;
}

bool Lidar::ReadImgDepth(const std::string& filename_diff) {
  Mat img_diff;
  if (!ReadH5(filename_diff, img_diff)) {
    cerr << "error in reading diff: " << filename_diff << endl;
    return false;
  }
  img_depth_ = img_diff;
  return true;
}

bool Lidar::ReadImgDepthObj(const std::string& filename) {
  std::ifstream ifstr;
  ifstr.open(filename.c_str());
  if (!ifstr.is_open()) {
    cerr << "Cannot open a file: " << filename << endl;
    return false;
  }
  std::string line;
  std::vector<Vector3d> vv;
  int h_max = 0;
  int w_max = 0;
  while (getline(ifstr, line)) {
    if (line == "")
      continue;
    std::istringstream isstr(line);
    std::string first_word;
    isstr >> first_word;
    if (first_word == "v") {
      Vector3d vertex;
      for (int i = 0; i < 3; ++i)
        isstr >> vertex[i];
      h_max = std::max(h_max, int(vertex[0]));
      w_max = std::max(w_max, int(vertex[1]));
      vv.push_back(vertex);
    }
  }
  ifstr.close();
  img_depth_.create(h_max + 1, w_max + 1, CV_32F);
  img_depth_.setTo(cv::Scalar(0));
  for (auto && v : vv) {
    img_depth_.at<float>(v[1], v[0]) = v[2];
  }
  return true;
}

bool Lidar::WriteImgDepthObj(const std::string& filename) const {
  Mat image_depth_cp = img_depth_.clone();
  std::vector<Vector3i> faces;
  for (int y = 1; y < image_depth_cp.rows - 1; ++y)
    for (int x = 1; x < image_depth_cp.cols - 1; ++x)
      image_depth_cp.at<float>(y, x) = img_depth_.at<float>(y, x) * 0.25 +
                                       img_depth_.at<float>(y - 1, x) * 0.125 + img_depth_.at<float>(y, x - 1) * 0.125 +
                                       img_depth_.at<float>(y + 1, x) * 0.125 + img_depth_.at<float>(y, x + 1) * 0.125 +
                                       img_depth_.at<float>(y - 1, x - 1) * 0.0625 + img_depth_.at<float>(y + 1, x - 1) * 0.0625 +
                                       img_depth_.at<float>(y + 1, x + 1) * 0.0625 + img_depth_.at<float>(y - 1, x + 1) * 0.0625 ;
  std::vector<Vector3d> vertices_pts;
  for (int y = 0; y < image_depth_cp.rows; ++y)
    for (int x = 0; x < image_depth_cp.cols; ++x)
      vertices_pts.emplace_back(Vector3d(x, y, image_depth_cp.at<float>(y, x)));

  int h = image_depth_cp.rows;
  int w = image_depth_cp.cols;
  for (int y = 0; y < image_depth_cp.rows - 1; ++y)
    for (int x = 0; x < image_depth_cp.cols - 1; ++x)
    {
      faces.emplace_back(Vector3i(y * w + x, y * w + x + 1, (y + 1)*w + x));
      faces.emplace_back(Vector3i((y + 1)*w + x + 1, y * w + x + 1, (y + 1)*w + x));
    }
  std::ofstream ofstr(filename);
  if (!ofstr.is_open())
    return false;
  for (const auto& point : vertices_pts)
    ofstr << "v " << point.transpose() << std::endl;
  for (const auto& face : faces)
    ofstr << "f " << face[0] + 1 << ' ' << face[1] + 1 << ' ' << face[2] + 1 << endl;
  ofstr.close();
}

bool Lidar::WriteImgDepth(const std::string & filename_out) {
  return WriteH5(filename_out, img_depth_);
}

bool Lidar::WriteImgNormal(const std::string & filename_out) {
  Mat img_normal;
  double kPixelsBetwPts = 1.0;
  ComputeNormalImg(img_depth_, kPixelsBetwPts, img_normal);
  cv::imwrite(filename_out, img_normal);
  return true;
}

bool Lidar::WritePointCloud(const std::string & filename_out) {
  std::vector<Vector3d> vertices_pts;
  for (int y = 0; y < img_depth_.rows; ++y)
    for (int x = 0; x < img_depth_.cols; ++x)
      vertices_pts.emplace_back(Vector3d(-1*x, y, img_depth_.at<float>(y, x)));
  std::ofstream ofstr(filename_out);
  if (!ofstr.is_open())
    return false;
  for (const auto& point : vertices_pts)
    ofstr << "v " << point.transpose() << std::endl;
  ofstr.close();
  return true;
}

bool Lidar::KeepCenterConnectedComponent() {
  Vector2i seed;
  FindCenterSeed(img_depth_, seed);
  int height = img_depth_.rows;
  int width = img_depth_.cols;
  if (img_depth_.at<float>(seed[0], seed[1]) > 0) {
    cv::Mat visited(height, width, CV_8UC1);
    visited.setTo(0);
    queue<Vector2i> bfsq;
    bfsq.push(Vector2i(seed[0], seed[1]));
    visited.at<uchar>(seed[0], seed[1]) = 255;
    std::vector<Vector2i> neighbors(4);
    neighbors[0] = Vector2i(0, 1); neighbors[1] = Vector2i(0, -1);
    neighbors[2] = Vector2i(-1, 0); neighbors[3] = Vector2i(1, 0);
    // BFS
    while (!bfsq.empty()) {
      Vector2i pos = bfsq.front();
      bfsq.pop();
      for (auto neighbor : neighbors) {
        Vector2i pos_child = pos + neighbor;
        if (pos_child[0] < 0 || pos_child[1] < 0 || pos_child[0] >= height || pos_child[1] >= width )
          continue;
        if (img_depth_.at<float>(pos_child[0], pos_child[1]) > 0 && !visited.at<uchar>(pos_child[0], pos_child[1])) {
          visited.at<uchar>(pos_child[0], pos_child[1]) = 255;
          bool flag = true;
          for (auto neighbor_c : neighbors) {
            Vector2i pos_grand_child = pos_child + neighbor_c;
            if (pos_grand_child[0] < 0 || pos_grand_child[1] < 0 || pos_grand_child[0] >= height || pos_grand_child[1] >= width )
              continue;
            if (img_depth_.at<float>(pos_grand_child[0], pos_grand_child[1]) == 0)
              flag = false;
          }
          if (flag == true) bfsq.push(pos_child);
        }
      }
    }
    img_depth_.setTo(0, visited == 0);
  }

  return true;
}

void Lidar::GetColoredDepthImage(cv::Mat & img_colored) {
  VisualizeDepthImage(img_depth_, img_colored);
}

bool Lidar::Recenteralize() {
  const double kMargin = 0.4;
  return DPM::Recenteralize(kMargin, img_depth_, offset_);
}


bool Lidar::AddNoise()
{
  // for (int j = 0; j < img_depth_.rows; ++j)
  //   for (int i = 0; i < img_depth_.cols; ++i)
  //     // if (rand() % 10 > 7 && img_depth_.at<float>(j, i) > 0)
  //     if (rand() % 10 > 3 && img_depth_.at<float>(j, i) > 0)
  //       // img_depth_.at<float>(j, i) += static_cast<double>(rand() % 3000 + 1 - 1500) / 1000;
  //       img_depth_.at<float>(j, i) += static_cast<double>(rand() % 3000 + 1 - 1500) / 1000;
  // return true;

  // Add gaussian noise
  const double mean = 0.0;
  const double stddev = 0.1;
  std::default_random_engine generator;
  std::normal_distribution<double> dist(mean, stddev);

  for (int j = 0; j < img_depth_.rows; ++j)
    for (int i = 0; i < img_depth_.cols; ++i)
      if (rand() % 10 > 3 && img_depth_.at<float>(j, i) > 0)
        img_depth_.at<float>(j, i) += dist(generator);
  return true;

}


} // DPM
