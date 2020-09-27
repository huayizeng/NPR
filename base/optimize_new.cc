#include "optimize_new.h"

#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <Eigen/Eigenvalues>
#include <limits>

#include "../rapidjson/document.h"
#include "../rapidjson/writer.h"
#include "../rapidjson/stringbuffer.h"
#include "../rapidjson/filereadstream.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "ceres/ceres.h"

#include "lidar.h"
#include "cad.h"
#include "image_process.h"

using Eigen::Vector2i;
using Eigen::Vector3i;
using Eigen::Vector2d;
using Eigen::Vector3d;

using cv::Mat;

using ceres::LossFunctionWrapper;
using ceres::NumericDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

using std::ifstream;
using std::ofstream;
using std::istringstream;
using std::numeric_limits;

namespace {

static const double PI = 3.1415927;
static const double sigma = 45; // best at this moment: 45

void RotateVertices(const float& angle, const Vector2d &vertex_ori, std::vector<Vector2d>& vertices) {
  float a = float(angle) / 360.0 * 2 * PI;
  float s = sin(a);
  float c = cos(a);
  std::for_each(vertices.begin(), vertices.end(), [s, c, &vertex_ori](Vector2d & vertex) {
Vector2d diff(vertex[0] - vertex_ori[0], vertex[1] - vertex_ori[1]);
vertex[0] = (diff[0] * c - diff[1] * s) + vertex_ori[0]; // x
vertex[1] = (1 * diff[0] * s + diff[1] * c) + vertex_ori[1]; // y
    });
}

template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  out << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i];
    if (i != v.size() - 1) out << ", ";
  }
  out << "]";
  return out;
}

double CombineResidualCauchy(const DPM::House& house) {
  DPM::Lidar lidar = house.get_lidar();
  Mat img_depth = lidar.get_img_depth();
  std::vector<int> x_request, y_request;
  for (unsigned j = 0; j < img_depth.rows; ++j)
    for (unsigned i = 0; i < img_depth.cols; ++i) {
      x_request.push_back(i); y_request.push_back(j);
    }
  Mat img_gradient;
  DPM::ComputeGradient(img_depth, img_gradient);
  std::vector<double> residual_at_request;
  house.ComputeDistance(x_request, y_request, residual_at_request);
  double sum = 0.0;
  double sigma2_inv = 1 / (sigma * sigma);
  for (unsigned i = 0; i < residual_at_request.size(); ++i) {
    double g = static_cast<double>(img_gradient.at<uchar>(i / img_gradient.rows, i % img_gradient.rows));
    double temp = residual_at_request[i] * exp(-0.5 * g * g * sigma2_inv);
    sum += 0.5 * log(temp * temp + 1);
  }
  return sum;
}

  // 0: std::vector<double> angles;
  // 1,2: std::vector<double> tls_y; std::vector<double> tls_x;
  // 3,4: std::vector<double> ratios_s; std::vector<double> ratios_e;
  // 5,6: std::vector<double> heis; std::vector<double> wids;
  // 7,8: std::vector<double> hs_ridge; std::vector<double> hs_eave;
  // 9: std::vector<double> angles_delta;

//10
void HouseMemberToParasI(const DPM::House& house, ParasModule& para_module) {
  DPM::IModule iModule = *dynamic_cast<DPM::IModule*>(house.get_cad().p_module_root_.get());
  // TODO(Huayi): test file should test if all values of para_module still exist after leaving function
  para_module.iflats_ = iModule.iflats_; para_module.lflats_ = iModule.lflats_;
  
  std::vector<double> temp_h_eave; temp_h_eave.push_back(iModule.height_eave_); para_module.params[8] = temp_h_eave;
  std::vector<double> temp_h_ridge; temp_h_ridge.push_back(iModule.height_ridge_); para_module.params[7] = temp_h_ridge;
  std::vector<double> temp_ratio_s; temp_ratio_s.push_back(iModule.ratio_pos_left_end_ridge_); para_module.params[3] = temp_ratio_s;
  std::vector<double> temp_ratio_e; temp_ratio_e.push_back(iModule.ratio_pos_right_end_ridge_); para_module.params[4] = temp_ratio_e;

  auto footprint_1 = iModule.footprint_;
  std::vector<double> temp_tl_x; temp_tl_x.push_back(footprint_1[0][0]); para_module.params[2] = temp_tl_x;
  std::vector<double> temp_tl_y; temp_tl_y.push_back(footprint_1[0][1]); para_module.params[1] = temp_tl_y;
  auto wid1 = footprint_1[1] - footprint_1[0];
  auto hei1 = (footprint_1[3] - footprint_1[0]);

  std::vector<double> temp_ratio_wid; temp_ratio_wid.push_back(wid1.norm()); para_module.params[6] = temp_ratio_wid;
  std::vector<double> temp_ratio_hei; temp_ratio_hei.push_back(hei1.norm()); para_module.params[5] = temp_ratio_hei;
  std::vector<double> temp_ratio_angle; temp_ratio_angle.push_back(atan2(wid1[1], wid1[0])); para_module.params[0] = temp_ratio_angle;
  std::vector<double> temp_ratio_angle_delta; 
  temp_ratio_angle_delta.push_back(atan2(hei1[1], hei1[0]) - atan2(wid1[1], wid1[0])); para_module.params[9] = temp_ratio_angle_delta;
  para_module.rooftypes_.push_back(static_cast<std::underlying_type<DPM::RoofType>::type>(iModule.get_rooftype()));
}

//13
void HouseMemberToParasL(const DPM::House& house, ParasModule& para_module) {
  // cerr << "into HouseMemberToParasL " << endl;
  DPM::LModule lModule = *dynamic_cast<DPM::LModule*>(house.get_cad().p_module_root_.get());
  para_module.iflats_ = lModule.iflats_; para_module.lflats_ = lModule.lflats_;

  std::vector<double> temp_h_eave; temp_h_eave.push_back(lModule.imodule_first_.height_eave_);
  para_module.params[8] = temp_h_eave;
  std::vector<double> temp_h_ridge; temp_h_ridge.push_back(lModule.imodule_first_.height_ridge_); 
  temp_h_ridge.push_back(lModule.imodule_second_.height_ridge_); para_module.params[7] = temp_h_ridge;
  std::vector<double> temp_ratio_e; temp_ratio_e.push_back(lModule.imodule_first_.ratio_pos_right_end_ridge_); 
  temp_ratio_e.push_back(lModule.imodule_second_.ratio_pos_right_end_ridge_); para_module.params[4] = temp_ratio_e;

  auto footprint_1 = lModule.imodule_first_.footprint_; auto footprint_2 = lModule.imodule_second_.footprint_;
  std::vector<double> temp_tl_x; temp_tl_x.push_back(footprint_1[0][0]); para_module.params[2] = temp_tl_x;
  std::vector<double> temp_tl_y; temp_tl_y.push_back(footprint_1[0][1]); para_module.params[1] = temp_tl_y;

  auto wid1 = footprint_1[1] - footprint_1[0]; auto hei1 = footprint_1[3] - footprint_1[0];
  auto wid2 = footprint_2[1] - footprint_2[0]; auto hei2 = footprint_2[3] - footprint_2[0];
  std::vector<double> temp_ratio_wid; temp_ratio_wid.push_back(wid1.norm()); temp_ratio_wid.push_back(wid2.norm()); para_module.params[6] = temp_ratio_wid;
  std::vector<double> temp_ratio_hei; temp_ratio_hei.push_back(hei1.norm()); temp_ratio_hei.push_back(hei2.norm()); para_module.params[5] = temp_ratio_hei;
  std::vector<double> temp_ratio_angle; temp_ratio_angle.push_back(atan2(wid1[1], wid1[0])); para_module.params[0] = temp_ratio_angle;
  std::vector<double> temp_ratio_angle_delta; 
  temp_ratio_angle_delta.push_back(atan2(hei1[1], hei1[0]) - atan2(wid1[1], wid1[0])); para_module.params[9] = temp_ratio_angle_delta;
  para_module.rooftypes_.push_back(static_cast<std::underlying_type<DPM::RoofType>::type>(lModule.imodule_first_.get_rooftype()));
  para_module.rooftypes_.push_back(static_cast<std::underlying_type<DPM::RoofType>::type>(lModule.imodule_second_.get_rooftype()));

}

//20
void HouseMemberToParasTwoI(const DPM::House& house, ParasModule& para_module) {

  // cerr << "into HouseMemberToParas2i " << endl;
  DPM::TwoIModule twoiModule = *dynamic_cast<DPM::TwoIModule*>(house.get_cad().p_module_root_.get());
  para_module.iflats_ = twoiModule.iflats_; para_module.lflats_ = twoiModule.lflats_;
  std::vector<double> temp_h_eave; temp_h_eave.push_back(twoiModule.imodule_first_.height_eave_);
  temp_h_eave.push_back(twoiModule.imodule_second_.height_eave_); para_module.params[8] = temp_h_eave;
  std::vector<double> temp_h_ridge; temp_h_ridge.push_back(twoiModule.imodule_first_.height_ridge_); 
  temp_h_ridge.push_back(twoiModule.imodule_second_.height_ridge_); para_module.params[7] = temp_h_ridge;
  std::vector<double> temp_ratio_e; temp_ratio_e.push_back(twoiModule.imodule_first_.ratio_pos_right_end_ridge_); 
  temp_ratio_e.push_back(twoiModule.imodule_second_.ratio_pos_right_end_ridge_); para_module.params[4] = temp_ratio_e;
  std::vector<double> temp_ratio_s; temp_ratio_s.push_back(twoiModule.imodule_first_.ratio_pos_left_end_ridge_); 
  temp_ratio_s.push_back(twoiModule.imodule_second_.ratio_pos_left_end_ridge_); para_module.params[3] = temp_ratio_s;
  auto footprint_1 = twoiModule.imodule_first_.footprint_; auto footprint_2 = twoiModule.imodule_second_.footprint_;
  std::vector<double> temp_tl_x; temp_tl_x.push_back(footprint_1[0][0]); temp_tl_x.push_back(footprint_2[0][0]); para_module.params[2] = temp_tl_x;
  std::vector<double> temp_tl_y; temp_tl_y.push_back(footprint_1[0][1]); temp_tl_y.push_back(footprint_2[0][1]); para_module.params[1] = temp_tl_y;

  auto wid1 = footprint_1[1] - footprint_1[0]; auto hei1 = footprint_1[3] - footprint_1[0];
  auto wid2 = footprint_2[1] - footprint_2[0]; auto hei2 = footprint_2[3] - footprint_2[0];
  std::vector<double> temp_ratio_wid; temp_ratio_wid.push_back(wid1.norm()); temp_ratio_wid.push_back(wid2.norm()); para_module.params[6] = temp_ratio_wid;
  std::vector<double> temp_ratio_hei; temp_ratio_hei.push_back(hei1.norm()); temp_ratio_hei.push_back(hei2.norm()); para_module.params[5] = temp_ratio_hei;
  std::vector<double> temp_ratio_angle; temp_ratio_angle.push_back(atan2(wid1[1], wid1[0])); para_module.params[0] = temp_ratio_angle;
  std::vector<double> temp_ratio_angle_delta; 
  temp_ratio_angle_delta.push_back(atan2(hei1[1], hei1[0]) - atan2(wid1[1], wid1[0])); 
  temp_ratio_angle_delta.push_back(atan2(wid2[1], wid2[0]) - atan2(wid1[1], wid1[0])); 
  temp_ratio_angle_delta.push_back(atan2(hei2[1], hei2[0]) - atan2(wid1[1], wid1[0])); 
  para_module.params[9] = temp_ratio_angle_delta;
  para_module.rooftypes_.push_back(static_cast<std::underlying_type<DPM::RoofType>::type>(twoiModule.imodule_first_.get_rooftype()));
  para_module.rooftypes_.push_back(static_cast<std::underlying_type<DPM::RoofType>::type>(twoiModule.imodule_second_.get_rooftype()));
}

//30
void HouseMemberToParasThreeI(const DPM::House& house, ParasModule& para_module) {
  DPM::PiModule threeiModule = *dynamic_cast<DPM::PiModule*>(house.get_cad().p_module_root_.get());
  para_module.iflats_ = threeiModule.iflats_; para_module.lflats_ = threeiModule.lflats_;

  std::vector<double> temp_h_eave; temp_h_eave.push_back(threeiModule.imodule_first_.height_eave_);
  temp_h_eave.push_back(threeiModule.imodule_second_.height_eave_); 
  temp_h_eave.push_back(threeiModule.imodule_third_.height_eave_); para_module.params[8] = temp_h_eave;
  std::vector<double> temp_h_ridge; temp_h_ridge.push_back(threeiModule.imodule_first_.height_ridge_); 
  temp_h_ridge.push_back(threeiModule.imodule_second_.height_ridge_); 
  temp_h_ridge.push_back(threeiModule.imodule_third_.height_ridge_); para_module.params[7] = temp_h_ridge;
  std::vector<double> temp_ratio_e; temp_ratio_e.push_back(threeiModule.imodule_first_.ratio_pos_right_end_ridge_); 
  temp_ratio_e.push_back(threeiModule.imodule_second_.ratio_pos_right_end_ridge_); 
  temp_ratio_e.push_back(threeiModule.imodule_third_.ratio_pos_right_end_ridge_); para_module.params[4] = temp_ratio_e;
  std::vector<double> temp_ratio_s; temp_ratio_s.push_back(threeiModule.imodule_first_.ratio_pos_left_end_ridge_); 
  temp_ratio_s.push_back(threeiModule.imodule_second_.ratio_pos_left_end_ridge_); 
  temp_ratio_s.push_back(threeiModule.imodule_third_.ratio_pos_left_end_ridge_); para_module.params[3] = temp_ratio_s;

  auto footprint_1 = threeiModule.imodule_first_.footprint_; auto footprint_2 = threeiModule.imodule_second_.footprint_; auto footprint_3 = threeiModule.imodule_third_.footprint_;
  std::vector<double> temp_tl_x; temp_tl_x.push_back(footprint_1[0][0]); temp_tl_x.push_back(footprint_2[0][0]); temp_tl_x.push_back(footprint_3[0][0]);para_module.params[2] = temp_tl_x;
  std::vector<double> temp_tl_y; temp_tl_y.push_back(footprint_1[0][1]); temp_tl_y.push_back(footprint_2[0][1]); temp_tl_y.push_back(footprint_3[0][1]);para_module.params[1] = temp_tl_y;

  auto wid1 = footprint_1[1] - footprint_1[0]; auto hei1 = footprint_1[3] - footprint_1[0];
  auto wid2 = footprint_2[1] - footprint_2[0]; auto hei2 = footprint_2[3] - footprint_2[0];
  auto wid3 = footprint_3[1] - footprint_3[0]; auto hei3 = footprint_3[3] - footprint_3[0];
  std::vector<double> temp_ratio_wid; temp_ratio_wid.push_back(wid1.norm()); temp_ratio_wid.push_back(wid2.norm()); temp_ratio_wid.push_back(wid3.norm()); para_module.params[6] = temp_ratio_wid;
  std::vector<double> temp_ratio_hei; temp_ratio_hei.push_back(hei1.norm()); temp_ratio_hei.push_back(hei2.norm()); temp_ratio_hei.push_back(hei3.norm()); para_module.params[5] = temp_ratio_hei;
  std::vector<double> temp_ratio_angle; temp_ratio_angle.push_back(atan2(wid1[1], wid1[0])); para_module.params[0] = temp_ratio_angle;
  std::vector<double> temp_ratio_angle_delta; 
  temp_ratio_angle_delta.push_back(atan2(hei1[1], hei1[0]) - atan2(wid1[1], wid1[0])); 
  temp_ratio_angle_delta.push_back(atan2(wid2[1], wid2[0]) - atan2(wid1[1], wid1[0])); 
  temp_ratio_angle_delta.push_back(atan2(hei2[1], hei2[0]) - atan2(wid1[1], wid1[0])); 
  temp_ratio_angle_delta.push_back(atan2(wid3[1], wid3[0]) - atan2(wid1[1], wid1[0])); 
  temp_ratio_angle_delta.push_back(atan2(hei3[1], hei3[0]) - atan2(wid1[1], wid1[0])); 
  para_module.params[9] = temp_ratio_angle_delta;
  para_module.rooftypes_.push_back(static_cast<std::underlying_type<DPM::RoofType>::type>(threeiModule.imodule_first_.get_rooftype()));
  para_module.rooftypes_.push_back(static_cast<std::underlying_type<DPM::RoofType>::type>(threeiModule.imodule_second_.get_rooftype()));
  para_module.rooftypes_.push_back(static_cast<std::underlying_type<DPM::RoofType>::type>(threeiModule.imodule_third_.get_rooftype()));
}

// 16
void HouseMemberToParasU(const DPM::House& house, ParasModule& para_module) {
  DPM::UModule uModule = *dynamic_cast<DPM::UModule*>(house.get_cad().p_module_root_.get());
  para_module.iflats_ = uModule.iflats_; para_module.lflats_ = uModule.lflats_;

  std::vector<double> temp_h_eave; temp_h_eave.push_back(uModule.imodule_first_.height_eave_);
  para_module.params[8] = temp_h_eave;
  std::vector<double> temp_h_ridge; temp_h_ridge.push_back(uModule.imodule_first_.height_ridge_); 
  temp_h_ridge.push_back(uModule.imodule_second_.height_ridge_); 
  temp_h_ridge.push_back(uModule.imodule_third_.height_ridge_); para_module.params[7] = temp_h_ridge;
  std::vector<double> temp_ratio_e; 
  temp_ratio_e.push_back(uModule.imodule_second_.ratio_pos_right_end_ridge_); 
  temp_ratio_e.push_back(uModule.imodule_third_.ratio_pos_right_end_ridge_); para_module.params[4] = temp_ratio_e;

  auto footprint_1 = uModule.imodule_first_.footprint_; auto footprint_2 = uModule.imodule_second_.footprint_; auto footprint_3 = uModule.imodule_third_.footprint_;
  std::vector<double> temp_tl_x; temp_tl_x.push_back(footprint_1[0][0]); para_module.params[2] = temp_tl_x;
  std::vector<double> temp_tl_y; temp_tl_y.push_back(footprint_1[0][1]); para_module.params[1] = temp_tl_y;

  auto wid1 = footprint_1[1] - footprint_1[0]; auto hei1 = footprint_1[3] - footprint_1[0];
  auto wid2 = footprint_2[1] - footprint_2[0]; auto hei2 = footprint_2[3] - footprint_2[0];
  auto wid3 = footprint_3[1] - footprint_3[0]; auto hei3 = footprint_3[3] - footprint_3[0];
  std::vector<double> temp_ratio_wid; temp_ratio_wid.push_back(wid1.norm()); temp_ratio_wid.push_back(wid2.norm()); temp_ratio_wid.push_back(wid3.norm()); para_module.params[6] = temp_ratio_wid;
  std::vector<double> temp_ratio_hei; temp_ratio_hei.push_back(hei1.norm()); temp_ratio_hei.push_back(hei2.norm()); temp_ratio_hei.push_back(hei3.norm()); para_module.params[5] = temp_ratio_hei;
  std::vector<double> temp_ratio_angle; temp_ratio_angle.push_back(atan2(wid1[1], wid1[0])); para_module.params[0] = temp_ratio_angle;
  std::vector<double> temp_ratio_angle_delta; 
  temp_ratio_angle_delta.push_back(atan2(hei1[1], hei1[0]) - atan2(wid1[1], wid1[0])); 
  para_module.params[9] = temp_ratio_angle_delta;
  para_module.rooftypes_.push_back(static_cast<std::underlying_type<DPM::RoofType>::type>(uModule.imodule_first_.get_rooftype()));
  para_module.rooftypes_.push_back(static_cast<std::underlying_type<DPM::RoofType>::type>(uModule.imodule_second_.get_rooftype()));
  para_module.rooftypes_.push_back(static_cast<std::underlying_type<DPM::RoofType>::type>(uModule.imodule_third_.get_rooftype()));

}

void SetIModule(const double angle, const double tl_x1, const double tl_y1, const double ratios_s, const double ratios_e, 
  const double hei1, const double wid1, const double h_ridge1, const double h_eave1, const double angle_delta1, const int rooftype1,
  DPM::IModule& iModule) {
  Vector2d v0 = Vector2d(tl_x1, tl_y1);
  Vector2d v1 = Vector2d(tl_x1 + wid1 * cos(angle), tl_y1 + wid1 * sin(angle));
  Vector2d v3 = Vector2d(tl_x1 + hei1 * cos(angle + angle_delta1), tl_y1 + hei1 * sin(angle + angle_delta1));
  Vector2d v2 = v3 - v0 + v1;
  iModule.footprint_.emplace_back(std::move(v0)); iModule.footprint_.emplace_back(std::move(v1));
  iModule.footprint_.emplace_back(std::move(v2)); iModule.footprint_.emplace_back(std::move(v3));
  iModule.height_eave_ = h_eave1;
  iModule.height_ridge_ = h_ridge1;
  iModule.ratio_pos_left_end_ridge_ = ratios_s;
  iModule.ratio_pos_right_end_ridge_ = ratios_e;
  iModule.azimuth_ = angle;
  iModule.rooftype_ = rooftype1 == 1 ? DPM::RoofType::kGable : DPM::RoofType::kHip; 
}

void ParasToHouseMemberI(const ParasModule& para_module, DPM::House& house) {
  DPM::IModule iModule;
  iModule.footprint_.clear();
  double angle = para_module.params[0][0];
  double tl_y1 = para_module.params[1][0]; double tl_x1 = para_module.params[2][0];
  double ratios_s = para_module.params[3][0]; double ratios_e = para_module.params[4][0];
  double hei1 = para_module.params[5][0]; double wid1 = para_module.params[6][0];
  double h_ridge1 = para_module.params[7][0]; double h_eave1 = para_module.params[8][0];
  double angle_delta1 = para_module.params[9][0];
  int rooftype1 = para_module.rooftypes_[0];
  SetIModule(angle, tl_x1, tl_y1, ratios_s, ratios_e, hei1, wid1, h_ridge1, h_eave1, angle_delta1, rooftype1, iModule);
  iModule.iflats_ = para_module.iflats_; iModule.lflats_ = para_module.lflats_;
  DPM::Cad cad;
  cad.p_module_root_ = std::make_shared<DPM::IModule>(iModule);
  cad.ReconstuctAll(false);
  house.set_cad(cad);
  // cad.WriteMesh("/Users/huayizeng/Desktop/temp.obj");
}

void ParasToHouseMemberL(const ParasModule& para_module, DPM::House& house) {
  DPM::LModule lModule;
  lModule.imodule_first_.footprint_.clear(); lModule.imodule_second_.footprint_.clear();

  double angle = para_module.params[0][0];
  double tl_y1 = para_module.params[1][0]; double tl_x1 = para_module.params[2][0];
  // double tl_y2 = para_module.params[1][1]; double tl_x2 = para_module.params[2][1];
  double ratio_e1 = para_module.params[4][0];double ratio_e2 = para_module.params[4][1];
  double hei1 = para_module.params[5][0]; double wid1 = para_module.params[6][0];
  double hei2 = para_module.params[5][1]; double wid2 = para_module.params[6][1];
  double h_ridge1 = para_module.params[7][0]; double h_eave1 = para_module.params[8][0];
  double h_ridge2 = para_module.params[7][1]; double h_eave2 = para_module.params[8][1];
  double angle_delta1 = para_module.params[9][0];
  int rooftype1 = para_module.rooftypes_[0]; int rooftype2 = para_module.rooftypes_[1];
  SetIModule(angle, tl_x1, tl_y1, 0.05, ratio_e1, hei1, wid1, h_ridge1, h_eave1, angle_delta1, rooftype1, lModule.imodule_first_);
  SetIModule(angle+angle_delta1, tl_x1, tl_y1, 0.05, ratio_e2, hei2, wid2, h_ridge2, h_eave1, -1*angle_delta1, rooftype2, lModule.imodule_second_);

  lModule.iflats_ = para_module.iflats_; lModule.lflats_ = para_module.lflats_;
  // lModule.imodule_second_.rooftype_ = DPM::RoofType(rooftype2);
  DPM::Cad cad;
  cad.p_module_root_ = std::make_shared<DPM::LModule>(lModule);
  cad.ReconstuctAll(true);
  house.set_cad(cad);
}

void ParasToHouseMemberTwoI(const ParasModule& para_module, DPM::House& house) {
  DPM::TwoIModule twoiModule;
  twoiModule.imodule_first_.footprint_.clear(); twoiModule.imodule_second_.footprint_.clear();

  double angle = para_module.params[0][0];
  double tl_y1 = para_module.params[1][0]; double tl_x1 = para_module.params[2][0];
  double tl_y2 = para_module.params[1][1]; double tl_x2 = para_module.params[2][1];
  double ratio_s1 = para_module.params[3][0];double ratio_s2 = para_module.params[3][1];
  double ratio_e1 = para_module.params[4][0];double ratio_e2 = para_module.params[4][1];
  double hei1 = para_module.params[5][0]; double wid1 = para_module.params[6][0];
  double hei2 = para_module.params[5][1]; double wid2 = para_module.params[6][1];
  double h_ridge1 = para_module.params[7][0]; double h_eave1 = para_module.params[8][0];
  double h_ridge2 = para_module.params[7][1]; double h_eave2 = para_module.params[8][1];
  double angle_delta1 = para_module.params[9][0]; double angle_delta2 = para_module.params[9][1]; double angle_delta3 = para_module.params[9][2];
  int rooftype1 = para_module.rooftypes_[0]; int rooftype2 = para_module.rooftypes_[1];

  SetIModule(angle, tl_x1, tl_y1, ratio_s1, ratio_e1, hei1, wid1, h_ridge1, h_eave1, angle_delta1, rooftype1, twoiModule.imodule_first_);
  SetIModule(angle+angle_delta2, tl_x2, tl_y2, ratio_s2, ratio_e2, hei2, wid2, h_ridge2, h_eave2, angle_delta3, rooftype2, twoiModule.imodule_second_);

  twoiModule.iflats_ = para_module.iflats_; twoiModule.lflats_ = para_module.lflats_;
  // lModule.imodule_second_.rooftype_ = DPM::RoofType(rooftype2);
  DPM::Cad cad;
  cad.p_module_root_ = std::make_shared<DPM::TwoIModule>(twoiModule);
  cad.ReconstuctAll(false);
  house.set_cad(cad);
}

void ParasToHouseMemberU(const ParasModule& para_module, DPM::House& house) {
  DPM::UModule uModule;
  uModule.imodule_first_.footprint_.clear(); uModule.imodule_second_.footprint_.clear(); uModule.imodule_third_.footprint_.clear();

  double angle = para_module.params[0][0];
  double tl_x1 = para_module.params[2][0]; 
  double tl_y1 = para_module.params[1][0]; 
  double ratio_e2 = para_module.params[4][0];double ratio_e3 = para_module.params[4][1];
  double hei1 = para_module.params[5][0]; double hei2 = para_module.params[5][1]; double hei3 = para_module.params[5][2]; 
  double wid1 = para_module.params[6][0]; double wid2 = para_module.params[6][1]; double wid3 = para_module.params[6][2];
  double h_ridge1 = para_module.params[7][0]; double h_ridge2 = para_module.params[7][1]; double h_ridge3 = para_module.params[7][2];
  double h_eave1 = para_module.params[8][0]; 
  double angle_delta1 = para_module.params[9][0];
  int rooftype1 = para_module.rooftypes_[0]; int rooftype2 = para_module.rooftypes_[1]; int rooftype3 = para_module.rooftypes_[2];

  SetIModule(angle, tl_x1, tl_y1, 0.05, 0.95, hei1, wid1, h_ridge1, h_eave1, angle_delta1, rooftype1, uModule.imodule_first_);
  SetIModule(angle+angle_delta1, tl_x1, tl_y1, 0.05, ratio_e2, hei2, wid2, h_ridge2, h_eave1, -1*angle_delta1, rooftype2, uModule.imodule_second_);
  SetIModule(angle+angle_delta1, tl_x1+wid1 * cos(angle), tl_y1+wid1 * sin(angle), 0.05, ratio_e3, hei3, wid3, h_ridge3, h_eave1, angle_delta1, rooftype3, uModule.imodule_third_);

  uModule.iflats_ = para_module.iflats_; uModule.lflats_ = para_module.lflats_;
  // uModule.imodule_second_.rooftype_ = DPM::RoofType(rooftype2);
  DPM::Cad cad;
  cad.p_module_root_ = std::make_shared<DPM::UModule>(uModule);
  cad.ReconstuctAll(true);
  house.set_cad(cad);
}

void ParasToHouseMemberThreeI(const ParasModule& para_module, DPM::House& house) {
  DPM::PiModule threeiModule;
  threeiModule.imodule_first_.footprint_.clear(); threeiModule.imodule_second_.footprint_.clear(); threeiModule.imodule_third_.footprint_.clear();

  double angle = para_module.params[0][0];
  double tl_x1 = para_module.params[2][0]; double tl_y1 = para_module.params[1][0]; 
  double tl_x2 = para_module.params[2][1]; double tl_y2 = para_module.params[1][1]; 
  double tl_x3 = para_module.params[2][2]; double tl_y3 = para_module.params[1][2]; 
  double ratio_e1 = para_module.params[4][0]; double ratio_e2 = para_module.params[4][1]; double ratio_e3 = para_module.params[4][2];
  double ratio_s1 = para_module.params[3][0]; double ratio_s2 = para_module.params[3][1]; double ratio_s3 = para_module.params[3][2];
  double hei1 = para_module.params[5][0]; double hei2 = para_module.params[5][1]; double hei3 = para_module.params[5][2]; 
  double wid1 = para_module.params[6][0]; double wid2 = para_module.params[6][1]; double wid3 = para_module.params[6][2];
  double h_ridge1 = para_module.params[7][0]; double h_ridge2 = para_module.params[7][1]; double h_ridge3 = para_module.params[7][2];
  double h_eave1 = para_module.params[8][0]; double h_eave2 = para_module.params[8][1]; double h_eave3 = para_module.params[8][2];
  double angle_delta1 = para_module.params[9][0]; 
  double angle_delta2 = para_module.params[9][1]; double angle_delta3 = para_module.params[9][2];
  double angle_delta4 = para_module.params[9][3]; double angle_delta5 = para_module.params[9][4];
  int rooftype1 = para_module.rooftypes_[0]; int rooftype2 = para_module.rooftypes_[1]; int rooftype3 = para_module.rooftypes_[2];

  SetIModule(angle, tl_x1, tl_y1, ratio_s1, ratio_e1, hei1, wid1, h_ridge1, h_eave1, angle_delta1, rooftype1, threeiModule.imodule_first_);
  SetIModule(angle+angle_delta2, tl_x2, tl_y2, ratio_s2, ratio_e2, hei2, wid2, h_ridge2, h_eave2, angle_delta3, rooftype2, threeiModule.imodule_second_);
  SetIModule(angle+angle_delta4, tl_x3, tl_y3, ratio_s3, ratio_e3, hei3, wid3, h_ridge3, h_eave3, angle_delta5, rooftype3, threeiModule.imodule_third_);

  threeiModule.iflats_ = para_module.iflats_; threeiModule.lflats_ = para_module.lflats_;
  // threeiModule.imodule_second_.rooftype_ = DPM::RoofType(rooftype2);
  DPM::Cad cad;
  cad.p_module_root_ = std::make_shared<DPM::PiModule>(threeiModule);
  cad.ReconstuctAll(false);
  house.set_cad(cad);
}

void HouseMemberToParas(const DPM::House& house, ParasModule& para_module) {
  para_module.moduletype = house.get_cad().get_module_type();
  switch(para_module.moduletype) {
    case DPM::ModuleType::kIModule:
      HouseMemberToParasI(house, para_module);
      break;
    case DPM::ModuleType::kLModule:
      HouseMemberToParasL(house, para_module);
      break;
    case DPM::ModuleType::kTwoIModule:
      HouseMemberToParasTwoI(house, para_module);
      break;
    case DPM::ModuleType::kUModule:
      HouseMemberToParasU(house, para_module);
      break;
    case DPM::ModuleType::kPiModule:
      HouseMemberToParasThreeI(house, para_module);
      break;
  }
}

void ParasToHouseMember(const ParasModule& para_module, DPM::House& house) {
  switch(para_module.moduletype) {
    case DPM::ModuleType::kIModule:
      ParasToHouseMemberI(para_module, house);
      break;
    case DPM::ModuleType::kLModule:
      ParasToHouseMemberL(para_module, house);
      break;
    case DPM::ModuleType::kTwoIModule:
      ParasToHouseMemberTwoI(para_module, house);
      break;
    case DPM::ModuleType::kUModule:
      ParasToHouseMemberU(para_module, house);
      break;
    case DPM::ModuleType::kPiModule:
      ParasToHouseMemberThreeI(para_module, house);
      break;
  }
}

struct OneLidarPointResidual {
  OneLidarPointResidual(const DPM::House& house, const std::vector<int>& type_paras)
    : house_(house),
      type_paras_(type_paras){
        DPM::Lidar lidar = house.get_lidar();
        Mat img_depth = lidar.get_img_depth();
        DPM::ComputeGradient(img_depth, img_gradient_);
        x_request_.clear(); y_request_.clear();
        for (unsigned j = 0; j < img_depth.rows; ++j) {
          for (unsigned i = 0; i < img_depth.cols; ++i) {
            x_request_.push_back(i); y_request_.push_back(j);
          }
        }
      }
  bool operator()(const double* const v_paras, double* residual) const {
    ParasModule para_module; para_module.init();
    HouseMemberToParas(house_, para_module); // Let para_module has roof_type values
    para_module.DoubleVectorToParas(type_paras_, v_paras);
    // para_module.moduletype = house_.get_cad().get_module_type();
    DPM::House house_changing_paras = house_;
    ParasToHouseMember(para_module, house_changing_paras);
    assert(!house_changing_paras.get_lidar().get_img_depth().empty() &&  "error, no lidar stored during optimization");
    std::vector<double> residual_at_request;
    // assert(x_request_.size() == y_request_.size() && "the size of x_request_ not equal to y_request_");
    house_changing_paras.ComputeDistance(x_request_, y_request_, residual_at_request);
    // DPM::Cad cad2 = house_changing_paras.get_cad();
    // cad2.WriteMesh("/Users/huayizeng/Desktop/temp/" + gen_random(10) + ".obj");

    double sigma2_inv = 1 / (sigma * sigma);
    // cv::imshow(" ", img_gradient_); cv::waitKey();
    double sum = 0.0;

    for (unsigned i = 0; i < x_request_.size(); ++i) {
      double g = static_cast<double>(img_gradient_.at<uchar>(i / img_gradient_.rows, i % img_gradient_.rows));
      double temp = residual_at_request[i] * exp(-0.5 * g * g * sigma2_inv);
      // if (temp > 1) residual[i] = sqrt(2 * temp - 1);
      // else if (temp < -1) residual[i] = -1 * sqrt(-1 - 2 * temp);
      // else residual[i] = temp;
      residual[i] = sqrt(log(1 + temp * temp));
      sum += residual[i];

    }
    return true;
  }
  
private:
  DPM::House house_;
  std::vector<int> x_request_;
  std::vector<int> y_request_;
  Mat img_gradient_;
  std::vector<int> type_paras_;
};

double OptimizeHouseIterateExhaustive(DPM::House& house) {
  const int k_samples = 5; const int k_subset = 4;
  ParasModule paras; paras.init();
  cerr << "start exhaustive search: " << endl;
  double energy_min = CombineResidualCauchy(house);
  cerr << "start_energy: " << energy_min << endl;
  HouseMemberToParas(house, paras);
  for (int ind_roll = 0; ind_roll < 2; ++ind_roll) {
    cerr << "ind_roll: " << ind_roll << endl;
    double energy_delta = energy_min;
    while (energy_delta > 1) {
      double energy_prev = energy_min;
      std::vector<ParasModule > samples_para_module;    
      paras.cartesian(k_samples, k_subset, samples_para_module);
      #pragma omp parallel for
      for (int i = 0; i < samples_para_module.size(); ++i) {
        DPM::House house_temp = house;
        ParasToHouseMember(samples_para_module[i], house_temp);
        double energy_temp = CombineResidualCauchy(house_temp);
        #pragma omp critical
        if (energy_temp < energy_min) {
          energy_min = energy_temp;
          paras = samples_para_module[i];
        }
      }
      cerr << "energy_min: " << energy_min << endl;
      energy_delta = energy_prev - energy_min;
    }
  }
  ParasToHouseMember(paras, house);
  return energy_min;
}

double OptimizeHouseCeres(DPM::House& house) {
  cerr << "start OptimizeHouseCeres: " << endl;
  ParasModule paras; paras.init();
  HouseMemberToParas(house, paras);
  Problem problem;
  std::vector<double> v_paras, lower_bound_v, upper_bound_v;
  std::vector<int> type_paras, constant_inds, lower_bound_inds, upper_bound_inds;
  paras.ParasToDoubleVector(type_paras, v_paras, constant_inds, upper_bound_inds, upper_bound_v, lower_bound_inds, lower_bound_v);
  int n_pixels = house.get_lidar().get_img_depth().cols * house.get_lidar().get_img_depth().rows;
  paras.moduletype = house.get_cad().get_module_type();
  CostFunction * cost_function;
  switch(paras.moduletype) {
    case DPM::ModuleType::kIModule:
      cost_function = new NumericDiffCostFunction<OneLidarPointResidual, ceres::RIDDERS, ceres::DYNAMIC, 10> (new OneLidarPointResidual(house, type_paras), ceres::TAKE_OWNERSHIP, n_pixels);
      break;
    case DPM::ModuleType::kLModule:
      cost_function = new NumericDiffCostFunction<OneLidarPointResidual, ceres::RIDDERS, ceres::DYNAMIC, 13> (new OneLidarPointResidual(house, type_paras), ceres::TAKE_OWNERSHIP, n_pixels);
      break;
    case DPM::ModuleType::kTwoIModule:
      cost_function = new NumericDiffCostFunction<OneLidarPointResidual, ceres::RIDDERS, ceres::DYNAMIC, 20> (new OneLidarPointResidual(house, type_paras), ceres::TAKE_OWNERSHIP, n_pixels);
      break;
    case DPM::ModuleType::kUModule:
      cost_function = new NumericDiffCostFunction<OneLidarPointResidual, ceres::RIDDERS, ceres::DYNAMIC, 16> (new OneLidarPointResidual(house, type_paras), ceres::TAKE_OWNERSHIP, n_pixels);
      break;
    case DPM::ModuleType::kPiModule:
      cost_function = new NumericDiffCostFunction<OneLidarPointResidual, ceres::RIDDERS, ceres::DYNAMIC, 30> (new OneLidarPointResidual(house, type_paras), ceres::TAKE_OWNERSHIP, n_pixels);
      break;
  }  
  problem.AddResidualBlock(cost_function, NULL, v_paras.data());
  for(unsigned i = 0 ; i < lower_bound_v.size(); ++i)
    problem.SetParameterLowerBound(v_paras.data(), lower_bound_inds[i], lower_bound_v[i]);
  for(unsigned i = 0; i < upper_bound_v.size(); ++i)
    problem.SetParameterUpperBound(v_paras.data(), upper_bound_inds[i], upper_bound_v[i]);
  ceres::SubsetParameterization *constant_transform_parameterization = NULL;
  constant_transform_parameterization = new ceres::SubsetParameterization(v_paras.size(), constant_inds);
  problem.SetParameterization(v_paras.data(), constant_transform_parameterization);
  Solver::Options options_solver;
  options_solver.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options_solver, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  // WARNING: Note here DoubleVectorToParas do not changes the value of i_flat_, l_flat_ and rooftype_v;
  paras.DoubleVectorToParas(type_paras, v_paras);
  // if(is_gable1 == true) paras_i.rooftype1 = 0; else paras_i.rooftype1 = 1;
  ParasToHouseMember(paras, house);
  return 0;
}

}; // end of empty namespace

namespace DPM {

bool RunOptHouses(HouseGroup & housegroup) {
  srand (3);
  size_t counter = 0;
  for (auto && house : housegroup.houses) {
    cerr << "house num: " << counter << endl;
    cerr << "house name: " << house.get_name() << endl;
    cerr << static_cast<std::underlying_type<DPM::ModuleType>::type>(house.get_cad().get_module_type()) << endl;
    auto type_temp = house.get_cad().get_module_type();
    // if(type_temp != DPM::ModuleType::kIModule) {
      OptimizeHouseIterateExhaustive(house);
    // }
    OptimizeHouseCeres(house);

    // TODO(H): adjut and see res
    // auto cad_temp = house.get_cad();
    // DPM::UModule uModule = *dynamic_cast<DPM::UModule*>(cad_temp.p_module_root_.get());
    // cerr << "uModule.imodule_second_.ratio_pos_right_end_ridge_: " << uModule.imodule_second_.ratio_pos_right_end_ridge_ << endl;
    // cerr << "uModule.imodule_third_.ratio_pos_right_end_ridge_: " << uModule.imodule_third_.ratio_pos_right_end_ridge_ << endl;
    // uModule.imodule_second_.ratio_pos_right_end_ridge_ = 1.0;
    // uModule.imodule_third_.ratio_pos_right_end_ridge_ = 1.0;
    // cad_temp.p_module_root_ = std::make_shared<DPM::UModule>(uModule);
    // cad_temp.ReconstuctAll();
    // house.set_cad(cad_temp);
    // //

    counter++;
  }
  return true;
}

void test();
static House generate_house_helper(const House& house, std::vector<DormerModule> dormers, const std::vector<int>& centers_inds){
  Cad tempcad = house.get_cad();
  std::shared_ptr<BaseModule> module_root = tempcad.p_module_root_->clone();
  module_root->assign_dormers_new(dormers, centers_inds);
  tempcad.p_module_root_ = module_root;
  bool result = tempcad.ReconstuctAll();
  House temp_house = house;
  if (result){
    temp_house.set_cad(tempcad);
  }
  return temp_house;
}
//static int written = 0;
void search_params_helper(std::vector<DormerParamSearching> &params,
                          int idx, std::vector<DormerModule> &results,
                          std::vector<DormerModule> &results_min,
                          const std::vector<Vector2d> &centers,
                          const std::vector<int>& centers_inds,
                          House &house,
                          double &energy_min){
  const DormerParamSearching &p = params[idx];
//  cerr << idx << endl;
  if (idx == -1){
    House temp_house = generate_house_helper(house, results, centers_inds);
    double energy_temp = CombineResidualCauchy(temp_house);
    // cerr << "energy_temp: " << energy_temp << endl;
    if (energy_temp < energy_min) {
      // cerr << energy_temp << endl;
      energy_min = energy_temp;
      results_min = results;
    }
    return;
  }
//  DormerModule tempDormer = results[p.dormer_no];
  switch (p.param) {
    case DormerParam::center_x_offset:
//      cerr << "center_x" << endl;
      for (int center_x_offset = -1; center_x_offset <= 1; ++center_x_offset) {
        int new_center_x = centers[p.dormer_no][0] + center_x_offset;
//        tempDormer.center[0] = new_center_x;
        results[p.dormer_no] = DormerModule( //TODO: bad code. Must fix!!
            Vector2d(new_center_x, results[p.dormer_no].center_[1]),
            results[p.dormer_no].radius_,
            results[p.dormer_no].triangular_ratio_,
            results[p.dormer_no].azimuth_,
            results[p.dormer_no].orig_,
            results[p.dormer_no].offset_,
//               const double ridge_degree,
            house
        );
        search_params_helper(params, idx-1, results, results_min, centers,centers_inds, house, energy_min);
      }
      break;
    case DormerParam::center_y_offset:
//      cerr << "center_y_offset" << endl;
      for (int center_y_offset = -1; center_y_offset <= 1; ++center_y_offset) {
        int new_center_y = centers[p.dormer_no][1] + center_y_offset;
        results[p.dormer_no] = DormerModule(
            Vector2d(results[p.dormer_no].center_[0], new_center_y),
            results[p.dormer_no].radius_,
            results[p.dormer_no].triangular_ratio_,
            results[p.dormer_no].azimuth_,
            results[p.dormer_no].orig_,
            results[p.dormer_no].offset_,
            house
        );
        search_params_helper(params, idx-1, results,results_min, centers,centers_inds,  house, energy_min);
      }
      break;
    case DormerParam::width:
      if(centers.size() < 3) {
        for (double width = 1; width < 6; ++width) {
          results[p.dormer_no] = DormerModule(
              results[p.dormer_no].center_,
              Vector2d(width, results[p.dormer_no].radius_[1]),
              results[p.dormer_no].triangular_ratio_,
              results[p.dormer_no].azimuth_,
              results[p.dormer_no].orig_,
              results[p.dormer_no].offset_,
              house
          );
          search_params_helper(params, idx-1, results,results_min, centers,centers_inds,  house, energy_min);
        }
      }
      else{
        for (double width = 1; width <= 2.5; width = width + 0.5) {
          results[p.dormer_no] = DormerModule(
              results[p.dormer_no].center_,
              Vector2d(width, results[p.dormer_no].radius_[1]),
              results[p.dormer_no].triangular_ratio_,
              results[p.dormer_no].azimuth_,
              results[p.dormer_no].orig_,
              results[p.dormer_no].offset_,
              house
          );
          search_params_helper(params, idx-1, results,results_min, centers,centers_inds,  house, energy_min);
        }
      }        
      break;
    case DormerParam::height:
//      cerr << "height" << endl;
      if(centers.size() < 3) {
        for (double height = 1; height < 6; ++height) {
  //        tempDormer.radius[1] = height;
          results[p.dormer_no] = DormerModule(
              results[p.dormer_no].center_,
              Vector2d(results[p.dormer_no].radius_[0], height),
              results[p.dormer_no].triangular_ratio_,
              results[p.dormer_no].azimuth_,
              results[p.dormer_no].orig_,
              results[p.dormer_no].offset_,
              house
          );
          search_params_helper(params, idx-1, results,results_min, centers,centers_inds,  house, energy_min);
        }
      }
      else {
        for (double height = 1; height <= 2.5; height = height + 0.5) {
          results[p.dormer_no] = DormerModule(
              results[p.dormer_no].center_,
              Vector2d(results[p.dormer_no].radius_[0], height),
              results[p.dormer_no].triangular_ratio_,
              results[p.dormer_no].azimuth_,
              results[p.dormer_no].orig_,
              results[p.dormer_no].offset_,
              house
          );
          search_params_helper(params, idx-1, results,results_min, centers,centers_inds,  house, energy_min);
        }
      }
      break;
    case DormerParam::triangular_ratio:
      if(centers.size() <= 2) {
        for (double triangular_ratio = 0; triangular_ratio <= 0.5; triangular_ratio = triangular_ratio + 0.1) {
          results[p.dormer_no] = DormerModule(
              results[p.dormer_no].center_,
              results[p.dormer_no].radius_,
              triangular_ratio,
              results[p.dormer_no].azimuth_,
              results[p.dormer_no].orig_,
              results[p.dormer_no].offset_,
              house
          );
          search_params_helper(params, idx-1, results,results_min, centers,centers_inds,  house, energy_min);
        }
      }
      else {
        for (double triangular_ratio = 0.2; triangular_ratio <= 0.6; triangular_ratio = triangular_ratio + 0.1) {
          results[p.dormer_no] = DormerModule(
              results[p.dormer_no].center_,
              results[p.dormer_no].radius_,
              triangular_ratio,
              results[p.dormer_no].azimuth_,
              results[p.dormer_no].orig_,
              results[p.dormer_no].offset_,
              house
          );
          search_params_helper(params, idx-1, results,results_min, centers,centers_inds,  house, energy_min);
        }
      }
      break;
  }
}

std::vector<ChimneyModule> OptimizeChimneyExhaustiveSearch(House &house, const std::vector<Vector2d> &centers, const std::vector<int>& centers_inds) {
    if(centers.size() < 0){
      return std::vector<ChimneyModule>();
    }
    Lidar lidar = house.get_lidar();
    Mat img_depth = lidar.get_img_depth();
    Mat img_gradient;
    ComputeGradient(img_depth, img_gradient);  
    Mat img_surface;
    house.ComputeImgSurface(img_surface);
    std::shared_ptr<BaseModule> two_i_real = house.get_cad().p_module_root_;
    double  azimuth = two_i_real->calc_azimuth();
    Vector2d orig((img_depth.cols-1)/2, (img_depth.rows-1)/2);
    std::vector<ChimneyModule> results;
    int radius_seed = rand() % 3;
    int radius = 0.7;
    switch(radius_seed) {
      case 1:
        radius = 1.15;
      case 2:
        radius = 1.3;
    }
    for (const Vector2d center: centers) {
      ChimneyModule chimney = ChimneyModule(
          Vector2d(center[0], center[1]),
          Vector2d(radius, radius),
          // Vector2d(1.15, 1.15),
          azimuth,
          orig,
          house);
      results.push_back(chimney);
    }
    return results;
}

std::vector<DormerModule> OptimizeDormerExhaustiveSearch(House &house, const std::vector<Vector2d> &centers, const std::vector<int>& centers_inds) {
  
    if(centers.size() < 0){
      return std::vector<DormerModule>();
    }
    Lidar lidar = house.get_lidar();
    Mat img_depth = lidar.get_img_depth();
    Mat img_gradient;
    ComputeGradient(img_depth, img_gradient);  
    Mat img_surface;
    house.ComputeImgSurface(img_surface);
    std::shared_ptr<BaseModule> two_i_real = house.get_cad().p_module_root_;
    double  azimuth = two_i_real->calc_azimuth();
    // double azimuth = house.get_feature_dnn().azimuth();
    // cerr << "azimuthazimuthazimuth: " << azimuth << endl;
    Vector2d orig((img_depth.cols-1)/2, (img_depth.rows-1)/2);
    std::vector<DormerModule> results;
    
    // TODO(H): for debug
    for (const Vector2d center: centers) {
      DormerModule dormer = DormerModule(
          Vector2d(center[0], center[1]),
          Vector2d(1.6, 1.6),
          0.3,
          azimuth,
          orig,
          Vector2d(0, 0),
          house);
      results.push_back(dormer);
    }


    // double energy_min = CombineResidualCauchy(house);
    // cerr << "start_energy: " << energy_min << endl;
    // double energy_delta = energy_min;
    // const int kNumVariable = 6;
    // for (int retry = 0; retry < 1; ++retry) {
    //   cerr << "retry" << retry << endl;
    //   energy_delta = energy_min;
    //   while (energy_delta > 1) {
    //     double energy_prev = energy_min;
    //     std::vector<DormerParamSearching> params;
    //     for (int i = 0; i < centers.size(); i++) {
    //       for (int j = 0; j < 5; j++) {
    //         DormerParamSearching p = DormerParamSearching{
    //             dormer_no: i,
    //             param: static_cast<DormerParam>(j)
    //         };
    //         params.push_back(p);
    //       }
    //     }
    //     std::random_shuffle(params.begin(), params.end()); //TODO: CHange BACK!!!!
    //     params.resize(kNumVariable);
    //     std::vector<DormerModule> results_temp = results;
    //     search_params_helper(params,
    //                          params.size()-1,
    //                           results_temp,
    //                           results,
    //                           centers,
    //                           centers_inds,
    //                           house,
    //                           energy_min);
    //     Cad tempcad = house.get_cad();
    //     std::shared_ptr<BaseModule> two_i = tempcad.p_module_root_->clone();
    //     two_i->assign_dormers_new(results, centers_inds);
    //     tempcad.p_module_root_ = two_i;
    //     tempcad.ReconstuctAll();
    //     House temp_house = house;
    //     temp_house.set_cad(tempcad);
    //     double verify_min = CombineResidualCauchy(temp_house);
    //     cerr << "verify min" << verify_min << endl;
    //     cerr << "new energy: "<< energy_min << endl;
    //     energy_delta = energy_prev - energy_min;
    //   }
    // }


    return results;
  }

  void assignDormerHelper(DormerModule &dor, Cad &cad_temp){
    std::vector<DormerModule> dormers;
    dormers.push_back(dor);
    std::shared_ptr<BaseModule> two_i = cad_temp.p_module_root_;
    // two_i->assign_dormers(dormers);
    cad_temp.ReconstuctAll();
  }
  void assignDormersHelper(std::vector<DormerModule> &dor, Cad &cad_temp, const std::vector<int>& centers_inds){
    std::shared_ptr<BaseModule> two_i = cad_temp.p_module_root_;
    two_i->assign_dormers_new(dor, centers_inds);
    cad_temp.ReconstuctAll();
  }

  void assignChimneysHelper(std::vector<ChimneyModule> &dor, Cad &cad_temp, const std::vector<int>& centers_inds){
    std::shared_ptr<BaseModule> two_i = cad_temp.p_module_root_;
    two_i->assign_chimneys_new(dor, centers_inds);
    cad_temp.ReconstuctAll();
  }


bool RunOptDormers(HouseGroup & housegroup, FileIO &file_io){
  // #pragma omp parallel for
  for (size_t i = 0; i < housegroup.houses.size(); i++) {
    cerr << "house name: " << housegroup.houses[i].get_name() << endl;
    cerr << "house num: " << i << endl;
    cerr << "Optimizing Dormers..." << endl;
      Cad cad = housegroup.houses[i].get_cad();
      std::shared_ptr<BaseModule> module_root = housegroup.houses[i].get_cad().p_module_root_;
      auto dnn = housegroup.houses[i].get_feature_dnn();
      auto size = housegroup.houses[i].get_feature_dnn().dormer_x_size();
      std::vector<Vector2d> centers;
      std::vector<int> centers_inds;
      cerr << "dormer_index.size: " << dnn.dormer_index_size() << endl;
      cerr << "#dormers: " << size << endl;

      for(int i = 0; i < size; i++) {
        centers.push_back(Vector2d(dnn.dormer_x(i), dnn.dormer_y(i)));
        centers_inds.push_back(dnn.dormer_index(i));
      }
      cerr << "centers.size(): " << centers.size() << endl;
      // for(auto&& ele : centers) 
      //   cerr << ele.transpose() << endl;
      cerr << "centers_inds: " << centers_inds << endl;

      Vector2d vertex_ori_64(32, 32);
      RotateVertices(dnn.azimuth(), vertex_ori_64, centers);
      float k_scale = static_cast<float>(housegroup.houses[i].get_lidar().get_img_depth().cols) / 64;
      for(auto&& v : centers) {
        v[0] *= k_scale; v[1] *= k_scale;
      }
      if(centers.size() > 0){
        std::vector<DormerModule> result = OptimizeDormerExhaustiveSearch(housegroup.houses[i], centers, centers_inds);
        // for(const auto &d: result){
        //   std::cout << "result dormer: " << d.center << ", " << d.radius << ", " << d.triangular_ratio << std::endl;
        // }
        assignDormersHelper(result, cad, centers_inds);
        housegroup.houses[i].set_cad(cad);
      }
    }
    return true;
  }

bool RunOptChimneys(HouseGroup & housegroup, FileIO &file_io){
  #pragma omp parallel for
  for (size_t i = 0; i < housegroup.houses.size(); i++) {
    cerr << "house name: " << housegroup.houses[i].get_name() << endl;
    cerr << "house num: " << i << endl;
    cerr << "Optimizing Chimneys..." << endl;
      Cad cad = housegroup.houses[i].get_cad();
      std::shared_ptr<BaseModule> module_root = housegroup.houses[i].get_cad().p_module_root_;
      auto dnn = housegroup.houses[i].get_feature_dnn();
      auto size = housegroup.houses[i].get_feature_dnn().chimney_x_size();
      std::vector<Vector2d> centers;
      std::vector<int> centers_inds;
      for(int i = 0; i < size; i++) {
        centers.push_back(Vector2d(dnn.chimney_x(i), dnn.chimney_y(i)));
        centers_inds.push_back(dnn.chimney_index(i));
      }
      cerr << "Chimney centers_inds: " << centers_inds << endl;
      Vector2d vertex_ori_64(32, 32);
      RotateVertices(dnn.azimuth(), vertex_ori_64, centers);
      float k_scale = static_cast<float>(housegroup.houses[i].get_lidar().get_img_depth().cols) / 64;
      for(auto&& v : centers) {
        v[0] *= k_scale; v[1] *= k_scale;
      }
      if(centers.size() > 0){
        std::vector<ChimneyModule> result = OptimizeChimneyExhaustiveSearch(housegroup.houses[i], centers, centers_inds);
        assignChimneysHelper(result, cad, centers_inds);
        housegroup.houses[i].set_cad(cad);
      }
    }
    return true;
}

} // DPM
