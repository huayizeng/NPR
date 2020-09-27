#include "cad.h"
#include <Eigen/Dense>
#include <limits>
#include <algorithm>
#include <numeric>
#include <memory>
#include <vector>
#include <queue>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "image_process.h"
#include "util.h"

using std::cerr;
using std::endl;
using std::numeric_limits;

using Eigen::Vector2i;
using Eigen::Vector3i;
using Eigen::Vector2d;
using Eigen::Vector3d;
using cv::Mat;

namespace {

  static const std::string kTypeNames[] = {"Null", "False", "True", "Object", "Array", "String", "Number"};
  static const float PI = 3.1415927;

  inline double CCW(const Vector2d &p, const Vector2d &q, const Vector2d &s) {
    return (p[0] * q[1] - p[1] * q[0] + q[0] * s[1] - q[1] * s[0] + s[0] * p[1] - s[1] * p[0]);
  }

  inline bool InsideTriangles2d(const Vector2d& v, const Vector2d& v0, const Vector2d& v1, const Vector2d& v2) {
    float a = CCW(v, v0, v1); float b = CCW(v, v1, v2); float c = CCW(v, v2, v0);
    return ( a * b > 0 && b * c > 0 );
  }

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
  bool GetHeightsInRectangle(const cv::Mat &img_depth,
			     std::vector<Eigen::Vector2d> &footprint_, std::vector<float>& v_h) {
    auto ab = footprint_[1] - footprint_[0];
    auto ad = footprint_[3] - footprint_[0];
    float ab2 = ab.dot(ab); float ad2 = ad.dot(ad);
    for (int y = 0; y < img_depth.cols; ++y)
      for (int x = 0; x < img_depth.rows; ++x) {
	if (img_depth.at<float>(y, x) > 0 &&
            ab.dot(Vector2d(x, y) - footprint_[0]) >= 0 && ab.dot(Vector2d(x, y) - footprint_[0]) <= ab2 &&
            ad.dot(Vector2d(x, y) - footprint_[0]) >= 0 && ad.dot(Vector2d(x, y) - footprint_[0]) <= ad2)
	  v_h.push_back(img_depth.at<float>(y, x));
      }
    return true;
  }

  bool ComputeIParasHack(const cv::Mat &img_depth,
			 std::vector<Eigen::Vector2d> &footprint_,
			 float &height_eave_,
			 float &height_ridge_) {
    std::vector<float> v_h;
    GetHeightsInRectangle(img_depth, footprint_, v_h);
    if (v_h.size() < 2) {
      cerr << "too small size remained, throw this example" << endl;
      return false;
    }
    std::partial_sort(v_h.begin(), v_h.begin() + v_h.size() / 20 + 2, v_h.end());
    float h_lowest = std::accumulate(v_h.begin(), v_h.begin() + v_h.size() / 10 + 2, 0.0) / (v_h.size() / 10 + 1);
    std::partial_sort(v_h.begin(), v_h.begin() + v_h.size() / 20 + 2, v_h.end(), std::greater<float>());
    float h_highest = std::accumulate(v_h.begin(), v_h.begin() + v_h.size() / 20 + 2, 0.0) / (v_h.size() / 20 + 1);
    height_eave_ = h_lowest; height_ridge_ = h_highest - h_lowest;

    return true;
  }

  bool overlap(Vector3i &tri1, Vector3i &tri2) {
    int count = 0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
	if (tri1[i] == tri2[j])
	  count++;
    return count == 2 ? true : false;
  }

  static Eigen::Vector2d rotate_pnt(const Vector2d &pnt, const Vector2d &orig, const double azimuth) {
    const double PI = 3.141592653589793238463;
    float a = float(azimuth) / 360.0 * 2 * PI;
    float s = sin(a);
    float c = cos(a);
    Vector2d diff(pnt - orig);
    Eigen::Vector2d result_v;
    result_v[0] = (diff[0] * c - diff[1] * s) + orig[0]; // x
    result_v[1] = (1 * diff[0] * s + diff[1] * c) + orig[1]; // y
    return result_v;
  }

  void RecomputeFootprint(const Vector2d& center, const Vector2d& orig, const Vector2d& radius, const double azimuth, std::vector<Eigen::Vector2d>& footprint) {
    const double PI = 3.141592653589793238462643383279502884197169399375;
    // cerr << "azimuth in DormerModule::recomputeFootPrint: " << azimuth << endl;
    double azimuth_degree = azimuth * 180 / PI;
    // azimuth_degree = 0;
    // double azimuth_degree = azimuth;
    Vector2d center_r = rotate_pnt(center.cast<double>(), orig, -azimuth_degree);
    Vector2d pnt1_r = center_r + Vector2d(-radius[0], -radius[1]);
    Vector2d pnt2_r = center_r + Vector2d(radius[0], -radius[1]);
    Vector2d pnt3_r = center_r + Vector2d(radius[0], radius[1]);
    Vector2d pnt4_r = center_r + Vector2d(-radius[0], radius[1]);

    Vector2d pnt1 = rotate_pnt(pnt1_r, orig, azimuth_degree);
    Vector2d pnt2 = rotate_pnt(pnt2_r, orig, azimuth_degree);
    Vector2d pnt3 = rotate_pnt(pnt3_r, orig, azimuth_degree);
    Vector2d pnt4 = rotate_pnt(pnt4_r, orig, azimuth_degree);
    footprint.push_back((pnt1));
    footprint.push_back((pnt2));
    footprint.push_back((pnt3));
    footprint.push_back((pnt4));
  }

  void ConstructWallMesh(const int n_footprint, DPM::Mesh& mesh) {
    for(unsigned i = 0; i < n_footprint; ++i) {
      mesh.faces.emplace_back(Vector3i(0+i, (1+i)%n_footprint, (0+i)%n_footprint + n_footprint)); mesh.faces.emplace_back(Vector3i((1+i)%n_footprint, (0+i)%n_footprint+n_footprint, (1+i)%n_footprint+n_footprint));    
    }
  }

} // empty namespace

namespace DPM {

  bool GetHeightsInL(const cv::Mat &img_depth,
		     std::vector<Eigen::Vector2d> &footprint_, std::vector<float>& v_h) {
    auto ab = footprint_[1] - footprint_[2];
    auto ad = footprint_[3] - footprint_[2];
    float ab2 = ab.dot(ab); float ad2 = ad.dot(ad);
    for (int y = 0; y < img_depth.cols; ++y)
      for (int x = 0; x < img_depth.rows; ++x) {
	if (img_depth.at<float>(y, x) > 0 &&
            ab.dot(Vector2d(x, y) - footprint_[2]) >= 0 && ab.dot(Vector2d(x, y) - footprint_[2]) <= ab2 &&
            ad.dot(Vector2d(x, y) - footprint_[2]) >= 0 && ad.dot(Vector2d(x, y) - footprint_[2]) <= ad2)
	  v_h.push_back(img_depth.at<float>(y, x));
      }
    if(footprint_.size() > 4) {
      auto ab_second = footprint_[3] - footprint_[4];
      auto ad_second = footprint_[5] - footprint_[4];
      float ab_second2 = ab_second.dot(ab_second); float ad_second2 = ad_second.dot(ad_second);
      for (int y = 0; y < img_depth.cols; ++y)
	for (int x = 0; x < img_depth.rows; ++x) {
	  if (img_depth.at<float>(y, x) > 0 &&
              ab_second.dot(Vector2d(x, y) - footprint_[4]) >= 0 && ab_second.dot(Vector2d(x, y) - footprint_[4]) <= ab_second2 &&
              ad_second.dot(Vector2d(x, y) - footprint_[4]) >= 0 && ad_second.dot(Vector2d(x, y) - footprint_[4]) <= ad2)
	    v_h.push_back(img_depth.at<float>(y, x));
	}
    }
    return true;
  }
  bool ComputeLFlatHeightHack(const cv::Mat &img_depth,
			      std::vector<Eigen::Vector2d> &footprint_,
			      float &height_) {
    std::vector<float> v_h;
    GetHeightsInL(img_depth, footprint_, v_h);
    if (v_h.size() < 2) {
      cerr << "too small size remained, throw this example" << endl;
      return false;
    }
    height_ = static_cast<float>(std::accumulate(v_h.begin(), v_h.end(), 0)) / v_h.size();
    return true;
  }

  template <class T>
  bool FlatRecognition(const int k_vertices, const OptRecog &opt_recog, const double azimuth_, const Vector2d& vertex_ori_64, 
		       const float k_scale, std::vector<T>& lflats_) {
    int n_l_flat = opt_recog.feature_dnn.flat_l_x_size() / k_vertices;
    if(k_vertices == 4) 
      n_l_flat = opt_recog.feature_dnn.flat_i_x_size() / k_vertices;

    for(unsigned ind_l_flat = 0; ind_l_flat < n_l_flat; ++ind_l_flat) {
      T l_flat_temp;
      if(k_vertices == 6) 
        for (int i = 0; i < k_vertices; ++i)
         l_flat_temp.footprint_.push_back(Vector2d(opt_recog.feature_dnn.flat_l_x(ind_l_flat*k_vertices+i), opt_recog.feature_dnn.flat_l_y(ind_l_flat*k_vertices+i)));
      else
        for (int i = 0; i < k_vertices; ++i)
         l_flat_temp.footprint_.push_back(Vector2d(opt_recog.feature_dnn.flat_i_x(ind_l_flat*k_vertices+i), opt_recog.feature_dnn.flat_i_y(ind_l_flat*k_vertices+i)));
      l_flat_temp.azimuth_ = azimuth_;
      RotateVertices(azimuth_, vertex_ori_64, l_flat_temp.footprint_);
      for(auto&& v : l_flat_temp.footprint_) {
        v[0] *= k_scale; v[1] *= k_scale;
      }
      if(!ComputeLFlatHeightHack(opt_recog.img_depth, l_flat_temp.footprint_, l_flat_temp.height_))
        return false;
      lflats_.push_back(l_flat_temp);
    }
  }

  bool Cad::Recognition(const OptRecog &opt_recog) {
    // static_cast<std::underlying_type<ModuleType>::type>
    auto house_type = opt_recog.feature_dnn.housetype();
    // cerr << "name: " << opt_recog.feature_dnn.name() << endl;  
    // cerr << "house_type: " << house_type << endl;
    if(house_type == 1)
      p_module_root_ = std::make_shared<TwoIModule>();
    if(house_type == 2)
      p_module_root_ = std::make_shared<CompModule>();
    if(house_type == 3)
      p_module_root_ = std::make_shared<IModule>();
    if(house_type == 4)
      p_module_root_ = std::make_shared<PiModule>();
    if(house_type == 5)
      p_module_root_ = std::make_shared<LModule>();
    if(house_type == 6)
      p_module_root_ = std::make_shared<UModule>();
    if (!p_module_root_->ModuleRecognition(opt_recog))
      return false;
    return true;
  }

  bool IModule::ModuleRecognition(const OptRecog &opt_recog) {
    Vector2d vertex_ori(opt_recog.img_depth.cols / 2, opt_recog.img_depth.rows / 2);
    Vector2d vertex_ori_64(32, 32);
    azimuth_ = opt_recog.feature_dnn.azimuth();
    ratio_pos_left_end_ridge_ = 0.0; ratio_pos_right_end_ridge_ = 1.0;
    if(opt_recog.feature_dnn.rooftype_size() == 0)
      rooftype_ = RoofType::kHip; 
    else
      rooftype_ = opt_recog.feature_dnn.rooftype(0) == 1 ? RoofType::kGable : RoofType::kHip; 
    if(opt_recog.feature_dnn.footprint_x_size() != 4 || opt_recog.feature_dnn.footprint_y_size() != 4 )
      return false;
    for (int i = 0; i < 4; ++i)
      footprint_.push_back(Vector2d(opt_recog.feature_dnn.footprint_x(i), opt_recog.feature_dnn.footprint_y(i)));
    for(auto&& ele : footprint_)
      if(ele[0] < 0) 
    return false;
    RotateVertices(azimuth_, vertex_ori_64, footprint_);
    float k_scale = static_cast<float>(opt_recog.img_depth.cols) / 64;
    for(auto&& v : footprint_) {
      v[0] *= k_scale; v[1] *= k_scale;
    }
    if(!ComputeIParasHack(opt_recog.img_depth, footprint_, height_eave_, height_ridge_))
      return false;
    FlatRecognition(6, opt_recog, azimuth_, vertex_ori_64, k_scale, lflats_);
    FlatRecognition(4, opt_recog, azimuth_, vertex_ori_64, k_scale, iflats_);
    return true;
  }
  bool DormerModule::ModuleRecognition(const OptRecog &opt_recog) {
    return false;
  }

  bool TwoIModule::ModuleRecognition(const OptRecog &opt_recog) {
    std::vector<Eigen::Vector2d> footprint_i_first_; std::vector<Eigen::Vector2d> footprint_i_second_;
    Vector2d vertex_ori(opt_recog.img_depth.cols / 2, opt_recog.img_depth.rows / 2);
    Vector2d vertex_ori_64(32, 32);
    azimuth_ = opt_recog.feature_dnn.azimuth();
    // cerr << opt_recog.feature_dnn.azimuth() << endl;
    if(opt_recog.feature_dnn.footprint_x_size() != 8 || opt_recog.feature_dnn.footprint_y_size() != 8 )
      return false;
    imodule_first_.ratio_pos_left_end_ridge_ = 0.0; imodule_first_.ratio_pos_right_end_ridge_ = 1.0;
    if(opt_recog.feature_dnn.rooftype_size() < 2) {
      imodule_first_.rooftype_ = RoofType::kHip; imodule_second_.rooftype_ = RoofType::kHip; 
    }
    else {
      imodule_first_.rooftype_ = opt_recog.feature_dnn.rooftype(0) == 1 ? RoofType::kGable : RoofType::kHip; 
      imodule_second_.rooftype_ = opt_recog.feature_dnn.rooftype(1) == 1 ? RoofType::kGable : RoofType::kHip; 
    }
    imodule_second_.ratio_pos_left_end_ridge_ = 0.0; imodule_second_.ratio_pos_right_end_ridge_ = 1.0;
    // TODO(H): D
    // imodule_first_.rooftype_ = RoofType::kGable; imodule_second_.rooftype_ = RoofType::kGable; 
    
    for (int i = 0; i < 4; ++i){
      footprint_i_first_.push_back(Vector2d(opt_recog.feature_dnn.footprint_x(i), opt_recog.feature_dnn.footprint_y(i)));
    }
    for (int i = 4; i < 8; ++i){
      footprint_i_second_.push_back(Vector2d(opt_recog.feature_dnn.footprint_x(i), opt_recog.feature_dnn.footprint_y(i)));
    }
    float k_scale = static_cast<float>(opt_recog.img_depth.cols) / 64;
    RotateVertices(azimuth_, vertex_ori_64, footprint_i_first_);
    RotateVertices(azimuth_, vertex_ori_64, footprint_i_second_);
    for(auto&& v : footprint_i_first_) {
      v[0] *= k_scale; v[1] *= k_scale;
    }
    for(auto&& v : footprint_i_second_) {
      v[0] *= k_scale; v[1] *= k_scale;
    }
    imodule_first_.set_footprint(footprint_i_first_);
    imodule_second_.set_footprint(footprint_i_second_); 
    if(!ComputeIParasHack(opt_recog.img_depth, footprint_i_first_, imodule_first_.height_eave_, imodule_first_.height_ridge_))
      return false;
    if(!ComputeIParasHack(opt_recog.img_depth, footprint_i_second_, imodule_second_.height_eave_, imodule_second_.height_ridge_))
      return false;
    FlatRecognition(6, opt_recog, azimuth_, vertex_ori_64, k_scale, lflats_);
    FlatRecognition(4, opt_recog, azimuth_, vertex_ori_64, k_scale, iflats_);

    return true;
  }

  bool PiModule::ModuleRecognition(const OptRecog &opt_recog) {
    std::vector<Eigen::Vector2d> footprint_i_first_; std::vector<Eigen::Vector2d> footprint_i_second_;
    std::vector<Eigen::Vector2d> footprint_i_third_;
    azimuth_ = opt_recog.feature_dnn.azimuth();
    if(opt_recog.feature_dnn.footprint_x_size() != 12 || opt_recog.feature_dnn.footprint_y_size() != 12 )
      return false;
    imodule_first_.ratio_pos_left_end_ridge_ = 0.0; imodule_first_.ratio_pos_right_end_ridge_ = 1.0;
    imodule_second_.ratio_pos_left_end_ridge_ = 0.0; imodule_second_.ratio_pos_right_end_ridge_ = 1.0;
    imodule_third_.ratio_pos_left_end_ridge_ = 0.0; imodule_third_.ratio_pos_right_end_ridge_ = 1.0;
    if(opt_recog.feature_dnn.rooftype_size() < 3) {
      imodule_first_.rooftype_ = RoofType::kHip; imodule_second_.rooftype_ = RoofType::kHip; imodule_third_.rooftype_ = RoofType::kHip; 
    }
    else {
      imodule_first_.rooftype_ = opt_recog.feature_dnn.rooftype(0) == 1 ? RoofType::kGable : RoofType::kHip; 
      imodule_second_.rooftype_ = opt_recog.feature_dnn.rooftype(1) == 1 ? RoofType::kGable : RoofType::kHip; 
      imodule_third_.rooftype_ = opt_recog.feature_dnn.rooftype(2) == 1 ? RoofType::kGable : RoofType::kHip; 
    }

    Vector2d vertex_ori(opt_recog.img_depth.cols / 2, opt_recog.img_depth.rows / 2);
    Vector2d vertex_ori_64(32, 32);

    for (int i = 0; i < 4; ++i)
      footprint_i_first_.push_back(Vector2d(opt_recog.feature_dnn.footprint_x(i), opt_recog.feature_dnn.footprint_y(i)));
    for (int i = 4; i < 8; ++i){
      footprint_i_second_.push_back(Vector2d(opt_recog.feature_dnn.footprint_x(i), opt_recog.feature_dnn.footprint_y(i)));
    }
    for (int i = 8; i < 12; ++i){
      footprint_i_third_.push_back(Vector2d(opt_recog.feature_dnn.footprint_x(i), opt_recog.feature_dnn.footprint_y(i)));
    }
    RotateVertices(azimuth_, vertex_ori_64, footprint_i_first_); RotateVertices(azimuth_, vertex_ori_64, footprint_i_second_);
    RotateVertices(azimuth_, vertex_ori_64, footprint_i_third_);

    float k_scale = static_cast<float>(opt_recog.img_depth.cols) / 64;
    for(auto&& v : footprint_i_first_) {
      v[0] *= k_scale; v[1] *= k_scale;
    }
    for(auto&& v : footprint_i_second_) {
      v[0] *= k_scale; v[1] *= k_scale;
    }
    for(auto&& v : footprint_i_third_) {
      v[0] *= k_scale; v[1] *= k_scale;
    }
    imodule_first_.set_footprint(footprint_i_first_); imodule_second_.set_footprint(footprint_i_second_);
    imodule_third_.set_footprint(footprint_i_third_);

    if(!ComputeIParasHack(opt_recog.img_depth, footprint_i_first_, imodule_first_.height_eave_, imodule_first_.height_ridge_))
      return false;
    if(!ComputeIParasHack(opt_recog.img_depth, footprint_i_second_, imodule_second_.height_eave_, imodule_second_.height_ridge_))
      return false;
    if(!ComputeIParasHack(opt_recog.img_depth, footprint_i_third_, imodule_third_.height_eave_, imodule_third_.height_ridge_))
      return false;

    FlatRecognition(6, opt_recog, azimuth_, vertex_ori_64, k_scale, lflats_);
    FlatRecognition(4, opt_recog, azimuth_, vertex_ori_64, k_scale, iflats_);
    return true;
  }

  bool LModule::ModuleRecognition(const OptRecog &opt_recog) {
    std::vector<Eigen::Vector2d> footprint_i_first_; std::vector<Eigen::Vector2d> footprint_i_second_;
    Vector2d vertex_ori(opt_recog.img_depth.cols / 2, opt_recog.img_depth.rows / 2);
    Vector2d vertex_ori_64(32, 32);
    azimuth_ = opt_recog.feature_dnn.azimuth();
    imodule_first_.ratio_pos_left_end_ridge_ = 0.0; imodule_first_.ratio_pos_right_end_ridge_ = 1.0;
    imodule_second_.ratio_pos_left_end_ridge_ = 0.0; imodule_second_.ratio_pos_right_end_ridge_ = 1.0;
    if(opt_recog.feature_dnn.rooftype_size() < 2) {
      imodule_first_.rooftype_ = RoofType::kHip; imodule_second_.rooftype_ = RoofType::kHip; 
    }
    else {
      imodule_first_.rooftype_ = opt_recog.feature_dnn.rooftype(0) == 1 ? RoofType::kGable : RoofType::kHip; 
      imodule_second_.rooftype_ = opt_recog.feature_dnn.rooftype(1) == 1 ? RoofType::kGable : RoofType::kHip; 
    }
    
    if(opt_recog.feature_dnn.footprint_x_size() != 8 || opt_recog.feature_dnn.footprint_y_size() != 8 )
      return false;

    for (int i = 0; i < 4; ++i)
      footprint_i_first_.push_back(Vector2d(opt_recog.feature_dnn.footprint_x(i), opt_recog.feature_dnn.footprint_y(i)));
    for (int i = 4; i < 8; ++i){
      footprint_i_second_.push_back(Vector2d(opt_recog.feature_dnn.footprint_x(i), opt_recog.feature_dnn.footprint_y(i)));
    }
    for(auto&& ele : footprint_i_first_)
      if(ele[0] < 0) 
	return false;
    for(auto&& ele : footprint_i_second_)
      if(ele[0] < 0) 
	return false;

    RotateVertices(azimuth_, vertex_ori_64, footprint_i_first_);
    RotateVertices(azimuth_, vertex_ori_64, footprint_i_second_);
    float k_scale = static_cast<float>(opt_recog.img_depth.cols) / 64;
    for(auto&& v : footprint_i_first_) {
      v[0] *= k_scale; v[1] *= k_scale;
    }
    for(auto&& v : footprint_i_second_) {
      v[0] *= k_scale; v[1] *= k_scale;
    }
    imodule_first_.set_footprint(footprint_i_first_);
    imodule_second_.set_footprint(footprint_i_second_);

    if(!ComputeIParasHack(opt_recog.img_depth, footprint_i_first_, imodule_first_.height_eave_, imodule_first_.height_ridge_))
      return false;
    if(!ComputeIParasHack(opt_recog.img_depth, footprint_i_second_, imodule_second_.height_eave_, imodule_second_.height_ridge_))
      return false;

    float h_eave_mean = (imodule_first_.height_eave_ + imodule_second_.height_eave_) * 0.5;
    imodule_first_.height_eave_ = h_eave_mean; imodule_second_.height_eave_ = h_eave_mean;
    FlatRecognition(6, opt_recog, azimuth_, vertex_ori_64, k_scale, lflats_);
    FlatRecognition(4, opt_recog, azimuth_, vertex_ori_64, k_scale, iflats_);

    return true;
  }
  double LModule::calc_azimuth() {
    return imodule_first_.calc_azimuth();
  }
  bool UModule::ModuleRecognition(const OptRecog &opt_recog) {
    std::vector<Eigen::Vector2d> footprint_i_first_; std::vector<Eigen::Vector2d> footprint_i_second_;
    std::vector<Eigen::Vector2d> footprint_i_third_;
    azimuth_ = opt_recog.feature_dnn.azimuth();
    imodule_first_.ratio_pos_left_end_ridge_ = 0.0; imodule_first_.ratio_pos_right_end_ridge_ = 1.0;
    imodule_second_.ratio_pos_left_end_ridge_ = 0.0; imodule_second_.ratio_pos_right_end_ridge_ = 1.0;
    imodule_third_.ratio_pos_left_end_ridge_ = 0.0; imodule_third_.ratio_pos_right_end_ridge_ = 1.0;

    if(opt_recog.feature_dnn.rooftype_size() < 3) {
      imodule_first_.rooftype_ = RoofType::kHip; imodule_second_.rooftype_ = RoofType::kHip; imodule_third_.rooftype_ = RoofType::kHip; 
    }
    else {
      imodule_first_.rooftype_ = opt_recog.feature_dnn.rooftype(0) == 1 ? RoofType::kGable : RoofType::kHip; 
      imodule_second_.rooftype_ = opt_recog.feature_dnn.rooftype(1) == 1 ? RoofType::kGable : RoofType::kHip; 
      imodule_third_.rooftype_ = opt_recog.feature_dnn.rooftype(2) == 1 ? RoofType::kGable : RoofType::kHip; 
    }
    Vector2d vertex_ori(opt_recog.img_depth.cols / 2, opt_recog.img_depth.rows / 2);
    Vector2d vertex_ori_64(32, 32);

    if(opt_recog.feature_dnn.footprint_x_size() != 12 || opt_recog.feature_dnn.footprint_y_size() != 12 )
      return false;
  
    for (int i = 0; i < 4; ++i)
      footprint_i_first_.push_back(Vector2d(opt_recog.feature_dnn.footprint_x(i), opt_recog.feature_dnn.footprint_y(i)));
    for (int i = 4; i < 8; ++i){
      footprint_i_second_.push_back(Vector2d(opt_recog.feature_dnn.footprint_x(i), opt_recog.feature_dnn.footprint_y(i)));
    }
    for (int i = 8; i < 12; ++i){
      footprint_i_third_.push_back(Vector2d(opt_recog.feature_dnn.footprint_x(i), opt_recog.feature_dnn.footprint_y(i)));
    }
    RotateVertices(azimuth_, vertex_ori_64, footprint_i_first_); RotateVertices(azimuth_, vertex_ori_64, footprint_i_second_);
    RotateVertices(azimuth_, vertex_ori_64, footprint_i_third_);

    float k_scale = static_cast<float>(opt_recog.img_depth.cols) / 64;
    for(auto&& v : footprint_i_first_) {
      v[0] *= k_scale; v[1] *= k_scale; 
    }
    for(auto&& v : footprint_i_second_) {
      v[0] *= k_scale; v[1] *= k_scale; 
    }
    for(auto&& v : footprint_i_third_) {
      v[0] *= k_scale; v[1] *= k_scale; 
    }
    imodule_first_.set_footprint(footprint_i_first_); imodule_second_.set_footprint(footprint_i_second_);
    imodule_third_.set_footprint(footprint_i_third_);

    if(!ComputeIParasHack(opt_recog.img_depth, footprint_i_first_, imodule_first_.height_eave_, imodule_first_.height_ridge_))
      return false;
    if(!ComputeIParasHack(opt_recog.img_depth, footprint_i_second_, imodule_second_.height_eave_, imodule_second_.height_ridge_))
      return false;
    if(!ComputeIParasHack(opt_recog.img_depth, footprint_i_third_, imodule_third_.height_eave_, imodule_third_.height_ridge_))
      return false;

    float h_eave_mean = (imodule_first_.height_eave_ + imodule_second_.height_eave_ + imodule_third_.height_eave_) * 0.333;
    imodule_first_.height_eave_ = h_eave_mean; imodule_second_.height_eave_ = h_eave_mean; imodule_third_.height_eave_ = h_eave_mean;

    FlatRecognition(6, opt_recog, azimuth_, vertex_ori_64, k_scale, lflats_);
    FlatRecognition(4, opt_recog, azimuth_, vertex_ori_64, k_scale, iflats_);
    return true;
  }
  double UModule::calc_azimuth() {
    return imodule_first_.calc_azimuth();
  }
  bool IFlat::Reconstruct(Mesh &mesh, bool LandU) {
    Mesh mesh_temp;
    std::vector<Vector3d> vertices_temp;
    vertices_temp.clear();
    std::vector<Vector2d> footprint_copy = footprint_;
    for (auto && v : footprint_copy)
      vertices_temp.push_back(Vector3d(v[0], v[1], height_));
    for (auto && v : footprint_copy)
      vertices_temp.push_back(Vector3d(v[0], v[1], 0.0));
    mesh_temp.vertices = vertices_temp;
    ConstructWallMesh(4, mesh_temp);
    mesh_temp.faces.emplace_back(Vector3i(0, 1, 2));
    mesh_temp.faces.emplace_back(Vector3i(0, 2, 3));
    mesh.Merge(mesh_temp);
    return true;
  }

  bool LFlat::Reconstruct(Mesh &mesh, bool LandU) {
    Mesh mesh_temp;
    std::vector<Vector3d> vertices_temp;
    vertices_temp.clear();
    std::vector<Vector2d> footprint_copy = footprint_;
    for (auto && v : footprint_copy)
      vertices_temp.push_back(Vector3d(v[0], v[1], height_));
    for (auto && v : footprint_copy)
      vertices_temp.push_back(Vector3d(v[0], v[1], 0.0));
    mesh_temp.vertices = vertices_temp;
    ConstructWallMesh(6, mesh_temp);
    mesh_temp.faces.emplace_back(Vector3i(0,1,2));
    mesh_temp.faces.emplace_back(Vector3i(0,2,3));
    mesh_temp.faces.emplace_back(Vector3i(0,3,4));
    mesh_temp.faces.emplace_back(Vector3i(0,4,5));
    mesh.Merge(mesh_temp);
    return true;
  }

  bool IModule::Reconstruct(Mesh &mesh, bool LandU) {
    Mesh mesh_temp;
    std::vector<Vector3d> vertices_temp;
    vertices_temp.clear();
    std::vector<Vector2d> footprint_copy = footprint_;
    if (!LandU)
      if ((footprint_copy[3] - footprint_copy[0]).norm() > (footprint_copy[1] - footprint_copy[0]).norm())
        std::rotate(footprint_copy.begin(), footprint_copy.begin() + 1, footprint_copy.end());
    for (auto && v : footprint_copy)
      vertices_temp.push_back(Vector3d(v[0], v[1], height_eave_));

    for (auto && v : footprint_copy)
      vertices_temp.push_back(Vector3d(v[0], v[1], 0.0));

    auto diff = footprint_copy[1] - footprint_copy[0];
    Vector2d leftridge_2d = Vector2d((footprint_copy[0] + footprint_copy[3]) / 2) + diff * ratio_pos_left_end_ridge_;
    Vector2d rightridge_2d = Vector2d((footprint_copy[0] + footprint_copy[3]) / 2) + diff * ratio_pos_right_end_ridge_;
    Vector3d leftridge(leftridge_2d[0], leftridge_2d[1], height_eave_ + height_ridge_);
    Vector3d rightridge(rightridge_2d[0], rightridge_2d[1], height_eave_ + height_ridge_);
    vertices_temp.push_back(leftridge);
    vertices_temp.push_back(rightridge);

    mesh_temp.vertices = vertices_temp;
    ConstructWallMesh(4, mesh_temp);
    mesh_temp.faces.emplace_back(Vector3i(8, 0, 1));
    mesh_temp.faces.emplace_back(Vector3i(8, 3, 2));
    mesh_temp.faces.emplace_back(Vector3i(8, 0, 3));
    mesh_temp.faces.emplace_back(Vector3i(9, 2, 1));
    mesh_temp.faces.emplace_back(Vector3i(9, 1, 8));
    mesh_temp.faces.emplace_back(Vector3i(9, 2, 8));

    mesh.Merge(mesh_temp);
    for(auto&& iflat : iflats_)
      iflat.Reconstruct(mesh, LandU);
    for(auto&& lflat : lflats_)
      lflat.Reconstruct(mesh, LandU);
 
    for (auto &dormer : dormers) {
      // dormer.set_paren(&imodule_second_);
      if(isIndependent_) {
        dormer.set_paren(this);
        dormer.Reconstruct(mesh, false);        
      }
    }
    for (auto &chimney : chimneys) {
      // dormer.set_paren(&imodule_second_);
      if(isIndependent_) {
        chimney.Reconstruct(mesh, false);        
      }
    }    
    return true;
  }
  bool IModule::constructDormerHeatmap(Mat &heatmap) {
    for(auto &dormer:dormers){
      dormer.constructDormerHeatmap(heatmap);
    }
    return true;
  }

  float ComputeIntersection(const Vector2d& p0, const Vector2d& v0, const Vector2d& p1, const Vector2d& v1) {
    float s, t;
    s = (-v0[1] * (p0[0] - p1[0]) + v0[0] * (p0[1] - p1[1])) / (-v1[0] * v0[1] + v0[0] * v1[1]);
    t = ( v1[0] * (p0[1] - p1[1]) - v1[1] * (p0[0] - p1[0])) / (-v1[0] * v0[1] + v0[0] * v1[1]);
    return t;
  }


  bool UModule::AdjustingEngPointsU( int ind_inner_end1,int ind_inner_end12,int ind_inner_end2, int ind_inner_end22, 
				     int ind_helper,int ind_helper2, int ind_helper3,
				     Mesh& mesh) {
    // int offset = size of dormers in first module * 10; (1 dormer: 10, 2 dormers: 20)
    auto vertices = mesh.vertices;
    std::vector<Vector2d> vertices_2d;
    for (auto && ele : vertices)
      vertices_2d.push_back(Vector2d(ele[0], ele[1]));

    double kDeltaHMinimum = 0.85;

    if (std::abs(mesh.vertices[ind_inner_end1][2] - mesh.vertices[ind_inner_end2][2]) < kDeltaHMinimum) {
      float t1 = ComputeIntersection(vertices_2d[ind_inner_end12], vertices_2d[ind_inner_end1] - vertices_2d[ind_inner_end12],
				     vertices_2d[ind_inner_end22], vertices_2d[ind_inner_end2] - vertices_2d[ind_inner_end22]);
      vertices_2d[ind_inner_end1] = (vertices_2d[ind_inner_end1] - vertices_2d[ind_inner_end12]) * t1 + vertices_2d[ind_inner_end12];
      float t2 = ComputeIntersection(vertices_2d[ind_inner_end22], vertices_2d[ind_inner_end2] - vertices_2d[ind_inner_end22],
				     vertices_2d[ind_inner_end12], vertices_2d[ind_inner_end1] - vertices_2d[ind_inner_end12]);
      vertices_2d[ind_inner_end2] = (vertices_2d[ind_inner_end2] - vertices_2d[ind_inner_end22]) * t2 + vertices_2d[ind_inner_end22];

      mesh.vertices[ind_inner_end1] = Vector3d(vertices_2d[ind_inner_end1][0], vertices_2d[ind_inner_end1][1], mesh.vertices[ind_inner_end1][2]);
      mesh.vertices[ind_inner_end2] = Vector3d(vertices_2d[ind_inner_end2][0], vertices_2d[ind_inner_end2][1], mesh.vertices[ind_inner_end2][2]);

      mesh.vertices[ind_inner_end1][2] = mesh.vertices[ind_inner_end2][2]; mesh.vertices[ind_inner_end12][2] = mesh.vertices[ind_inner_end22][2];
    }
    else {
      // ind_inner_end1 is lower
      if (mesh.vertices[ind_inner_end1][2] > mesh.vertices[ind_inner_end2][2]) {
      	std::swap(ind_inner_end1, ind_inner_end2); std::swap(ind_inner_end12, ind_inner_end22); ind_helper = ind_helper2;
      }
      float t1 = ComputeIntersection(vertices_2d[ind_inner_end12], vertices_2d[ind_inner_end1] - vertices_2d[ind_inner_end12],
				     vertices_2d[ind_inner_end22], vertices_2d[ind_inner_end2] - vertices_2d[ind_inner_end22]);
      vertices_2d[ind_inner_end1] = (vertices_2d[ind_inner_end1] - vertices_2d[ind_inner_end12]) * t1 + vertices_2d[ind_inner_end12];
      mesh.vertices[ind_inner_end1] = Vector3d(vertices_2d[ind_inner_end1][0], vertices_2d[ind_inner_end1][1], mesh.vertices[ind_inner_end1][2]);
      Vector2d p0, v0, p1, v1;
      double ratio = (mesh.vertices[ind_inner_end2][2] - mesh.vertices[0][2]) / (mesh.vertices[ind_inner_end1][2] - mesh.vertices[0][2]);      
      Vector2d mid = Vector2d( (0.5 * (mesh.vertices[ind_helper3] + mesh.vertices[ind_helper]))[0], (0.5 * (mesh.vertices[ind_helper3] + mesh.vertices[ind_helper]))[1] );
      // cerr << "ratio: " << ratio << endl;
      Vector2d temp = mid + (vertices_2d[ind_inner_end1] - mid) * ratio;
      mesh.vertices[ind_inner_end2] = Vector3d(temp[0], temp[1], mesh.vertices[ind_inner_end2][2]);
    }

    return true;
  }


  bool LModule::AdjustingEngPoints(Mesh& mesh) {
    // int offset = size of dormers in first module * 10; (1 dormer: 10, 2 dormers: 20)
    int offset = 0;//10*dormers;
    auto vertices = mesh.vertices;
    std::vector<Vector2d> vertices_2d;
    for (auto && ele : vertices)
      vertices_2d.push_back(Vector2d(ele[0], ele[1]));
    int ind_inner_end1 = 8;
    int ind_inner_end12 = 9;
    int ind_inner_end2 = 18 + offset;
    int ind_inner_end22 = 19 + offset;
    int ind_helper = 13 + offset;
    double kDeltaHMinimum = 0.85;

    if (std::abs(mesh.vertices[ind_inner_end1][2] - mesh.vertices[ind_inner_end2][2]) < kDeltaHMinimum) {
      float t1 = ComputeIntersection(vertices_2d[ind_inner_end12], vertices_2d[ind_inner_end1] - vertices_2d[ind_inner_end12],
				     vertices_2d[ind_inner_end22], vertices_2d[ind_inner_end2] - vertices_2d[ind_inner_end22]);
      vertices_2d[ind_inner_end1] = (vertices_2d[ind_inner_end1] - vertices_2d[ind_inner_end12]) * t1 + vertices_2d[ind_inner_end12];

      float t2 = ComputeIntersection(vertices_2d[ind_inner_end22], vertices_2d[ind_inner_end2] - vertices_2d[ind_inner_end22],
				     vertices_2d[ind_inner_end12], vertices_2d[ind_inner_end1] - vertices_2d[ind_inner_end12]);
      vertices_2d[ind_inner_end2] = (vertices_2d[ind_inner_end2] - vertices_2d[ind_inner_end22]) * t2 + vertices_2d[ind_inner_end22];

      mesh.vertices[ind_inner_end1] = Vector3d(vertices_2d[ind_inner_end1][0], vertices_2d[ind_inner_end1][1], mesh.vertices[ind_inner_end1][2]);
      mesh.vertices[ind_inner_end2] = Vector3d(vertices_2d[ind_inner_end2][0], vertices_2d[ind_inner_end2][1], mesh.vertices[ind_inner_end2][2]);

      mesh.vertices[ind_inner_end1][2] = mesh.vertices[ind_inner_end2][2]; mesh.vertices[ind_inner_end12][2] = mesh.vertices[ind_inner_end22][2];
    }
    else {
      // ind_inner_end1 is lower
      if (mesh.vertices[ind_inner_end1][2] > mesh.vertices[ind_inner_end2][2]) {
	std::swap(ind_inner_end1, ind_inner_end2); std::swap(ind_inner_end12, ind_inner_end22); ind_helper = 3;
      }
      float t1 = ComputeIntersection(vertices_2d[ind_inner_end12], vertices_2d[ind_inner_end1] - vertices_2d[ind_inner_end12],
				     vertices_2d[ind_inner_end22], vertices_2d[ind_inner_end2] - vertices_2d[ind_inner_end22]);
      vertices_2d[ind_inner_end1] = (vertices_2d[ind_inner_end1] - vertices_2d[ind_inner_end12]) * t1 + vertices_2d[ind_inner_end12];
      mesh.vertices[ind_inner_end1] = Vector3d(vertices_2d[ind_inner_end1][0], vertices_2d[ind_inner_end1][1], mesh.vertices[ind_inner_end1][2]);
      Vector2d p0, v0, p1, v1;
      double ratio = (mesh.vertices[ind_inner_end2][2] - mesh.vertices[0][2]) / (mesh.vertices[ind_inner_end1][2] - mesh.vertices[0][2]);      
      Vector2d mid = Vector2d( (0.5 * (mesh.vertices[0] + mesh.vertices[ind_helper]))[0], (0.5 * (mesh.vertices[0] + mesh.vertices[ind_helper]))[1] );
      // cerr << "ratio: " << ratio << endl;
      Vector2d temp = mid + (vertices_2d[ind_inner_end1] - mid) * ratio;
      mesh.vertices[ind_inner_end2] = Vector3d(temp[0], temp[1], mesh.vertices[ind_inner_end2][2]);
    }
    // TODO(Henry): if two heights are very close, simply snap them into one
    return true;
  }

  bool LModule::Reconstruct(Mesh &mesh, bool LandU) {
    imodule_first_.isIndependent_ = false;
    imodule_second_.isIndependent_ = false;
    imodule_first_.Reconstruct(mesh, LandU);
    imodule_second_.Reconstruct(mesh, LandU);
    AdjustingEngPoints(mesh);
    for(auto&& iflat : iflats_)
      iflat.Reconstruct(mesh, LandU);
    for(auto&& lflat : lflats_)
      lflat.Reconstruct(mesh, LandU);


    for (auto &dormer : imodule_first_.dormers) {
      dormer.set_paren(&imodule_first_);
      dormer.Reconstruct(mesh, LandU);
    }
    for (auto &dormer : imodule_second_.dormers) {
      dormer.set_paren(&imodule_second_);
      dormer.Reconstruct(mesh, LandU);
    }
    for (auto &chimney : chimneys) {
      chimney.Reconstruct(mesh, LandU);
    }
    // for (auto &chimney : imodule_first_.chimneys) 
    //   chimney.Reconstruct(mesh, LandU);
    // for (auto &chimney : imodule_second_.chimneys) 
    //   chimney.Reconstruct(mesh, LandU);
    return true;
  }

  bool UModule::Reconstruct(Mesh &mesh, bool LandU) {
    imodule_first_.isIndependent_ = false;
    imodule_second_.isIndependent_ = false;
    imodule_third_.isIndependent_ = false;
    imodule_first_.Reconstruct(mesh, LandU);
    imodule_second_.Reconstruct(mesh, LandU);
    imodule_third_.Reconstruct(mesh, LandU);
    int ind_inner_end1 = 8;
    int ind_inner_end12 = 9;
    int ind_inner_end2 = 18;
    int ind_inner_end22 = 19;
    int ind_helper = 13;
    int ind_helper2 = 3;
    int ind_helper3 = 0;
    AdjustingEngPointsU(ind_inner_end1, ind_inner_end12, ind_inner_end2, ind_inner_end22, ind_helper, ind_helper2, ind_helper3, mesh);
    ind_inner_end1 = 9;
    ind_inner_end12 = 8;
    ind_inner_end2 = 28;
    ind_inner_end22 = 29;
    ind_helper = 23;
    ind_helper2 = 2;
    ind_helper3 = 1;
    AdjustingEngPointsU(ind_inner_end2, ind_inner_end22, ind_inner_end1, ind_inner_end12, ind_helper2, ind_helper, ind_helper3, mesh);
    for(auto&& iflat : iflats_)
      iflat.Reconstruct(mesh, LandU);
    for(auto&& lflat : lflats_)
      lflat.Reconstruct(mesh, LandU);
    for (auto &dormer : imodule_first_.dormers) {
      dormer.set_paren(&imodule_first_);
      dormer.Reconstruct(mesh, LandU);
    }
    for (auto &dormer : imodule_second_.dormers) {
      dormer.set_paren(&imodule_second_);
      dormer.Reconstruct(mesh, LandU);
    }
    for (auto &dormer : imodule_third_.dormers) {
      dormer.set_paren(&imodule_third_);
      dormer.Reconstruct(mesh, LandU);
    }
    for (auto &chimney : chimneys) 
      chimney.Reconstruct(mesh, LandU);

    return true;
  }



  bool TwoIModule::Reconstruct(Mesh &mesh, bool LandU) {
    // Naturally we should get 8 vertices on the ground as the output of this point.Vector2d
    // For the simplicity, we choose not to do anything here.
    imodule_first_.isIndependent_ = false;
    imodule_second_.isIndependent_ = false;
    imodule_first_.Reconstruct(mesh, LandU);
    imodule_second_.Reconstruct(mesh, LandU);
    for(auto&& iflat : iflats_)
      iflat.Reconstruct(mesh, LandU);
    for(auto&& lflat : lflats_)
      lflat.Reconstruct(mesh, LandU);


    for (auto &dormer : imodule_first_.dormers) {
      dormer.set_paren(&imodule_first_);
      dormer.Reconstruct(mesh, LandU);
    }
    for (auto &dormer : imodule_second_.dormers) {
      dormer.set_paren(&imodule_second_);
      dormer.Reconstruct(mesh, LandU);
      // cerr << "reconstruct dormer in second i" << endl; 
    }
    for (auto &chimney : imodule_first_.chimneys) 
      chimney.Reconstruct(mesh, LandU);
    for (auto &chimney : imodule_second_.chimneys) 
      chimney.Reconstruct(mesh, LandU);

    return true;
  }

  bool PiModule::Reconstruct(Mesh &mesh, bool LandU) {
    // Naturally we should get 8 vertices on the ground as the output of this point.Vector2d
    // For the simplicity, we choose not to do anything here.
    imodule_first_.isIndependent_ = false;
    imodule_second_.isIndependent_ = false;
    imodule_third_.isIndependent_ = false;
    imodule_first_.Reconstruct(mesh, LandU);
    imodule_second_.Reconstruct(mesh, LandU);
    imodule_third_.Reconstruct(mesh, LandU);
    for(auto&& iflat : iflats_)
      iflat.Reconstruct(mesh, LandU);
    for(auto&& lflat : lflats_)
      lflat.Reconstruct(mesh, LandU);

    for (auto &dormer : imodule_first_.dormers) {
      dormer.set_paren(&imodule_first_);
      dormer.Reconstruct(mesh, LandU);
    }
    for (auto &dormer : imodule_second_.dormers) {
      dormer.set_paren(&imodule_second_);
      dormer.Reconstruct(mesh, LandU);
    }
    for (auto &dormer : imodule_third_.dormers) {
      dormer.set_paren(&imodule_third_);
      dormer.Reconstruct(mesh, LandU);
    }
    for (auto &chimney : imodule_first_.chimneys) 
      chimney.Reconstruct(mesh, LandU);
    for (auto &chimney : imodule_second_.chimneys) 
      chimney.Reconstruct(mesh, LandU);
    for (auto &chimney : imodule_third_.chimneys) 
      chimney.Reconstruct(mesh, LandU);
    return true;
  }

  double IModule::calc_azimuth() {
    const Vector2d line = footprint_[1] - footprint_[0];
    const Vector2d x_line(1, 0);
    //  double cosTheta = line.dot(x_line) / (line.norm() * x_line.norm());
    //  double angle = acos(cosTheta);
    double dot = line.dot(x_line);
    double det = line[0] * x_line[1] - x_line[0] * line[1];
    double angle = atan2(det, dot);
    const double angles[] = {-3.141592653589793, -1.570796326794897 , 0.0, 1.570796326794897, 3.141592653589793};
    std::vector<double> diff;
    for (int i = 0; i < sizeof(angles) / sizeof(double); i++) {
      diff.push_back(angle - angles[i]);
    }
    std::vector<double> abs_diff;
    for (int i = 0; i < diff.size(); i++) {
      abs_diff.push_back(std::abs(diff[i]));
    }
    size_t least_elem_idx = 0;
    double least_elem = std::numeric_limits<double>::max();
    for (int i = 0; i < abs_diff.size(); i++) {
      if (abs_diff[i] < least_elem) {
  	least_elem = abs_diff[i];
  	least_elem_idx = i;
      }
    }
    return -diff[least_elem_idx];
  }
  double TwoIModule::calc_azimuth(){
    return imodule_first_.calc_azimuth();
  }
  double PiModule::calc_azimuth() {
    const Vector2d line = imodule_first_.footprint_[1] - imodule_first_.footprint_[0];
    const Vector2d x_line(1, 0);
    double cosTheta = line.dot(x_line) / (line.norm() * x_line.norm());
    double angle = acos(cosTheta);
    const double angles[] = {0.0, 1.570796326794897, 3.141592653589793};
    std::vector<double> diff;
    for (int i = 0; i < sizeof(angles) / sizeof(double); i++) {
      diff.push_back(angle - angles[i]);
    }
    std::vector<double> abs_diff;
    for (int i = 0; i < diff.size(); i++) {
      abs_diff.push_back(std::abs(diff[i]));
    }
    size_t least_elem_idx = 0;
    double least_elem = std::numeric_limits<double>::max();
    for (int i = 0; i < abs_diff.size(); i++) {
      if (abs_diff[i] < least_elem) {
	least_elem = abs_diff[i];
	least_elem_idx = i;
      }
    }
    return diff[least_elem_idx];
  }

  static bool check_point_in_footprint(const std::vector<Eigen::Vector2d> &footprint, const Eigen::Vector2d &new_point) {
    Eigen::Vector2d a = footprint[0];
    Eigen::Vector2d b = footprint[1];
    Eigen::Vector2d c = footprint[2];
    Eigen::Vector2d ab = b - a;
    Eigen::Vector2d bc = c - b;
    Eigen::Vector2d am = new_point - a;
    Eigen::Vector2d bm = new_point - b;
    return (ab.dot(am) > 0) && (ab.dot(am) < ab.dot(ab)) && (bc.dot(bm) > 0) && (bc.dot(bm) < bc.dot(bc));
  }
  static double compute_distance(const Eigen::Vector2d &pnt,
				 const Eigen::Vector2d &pnt_on_line,
				 const Eigen::Vector2d &line) {
    return ((pnt_on_line - pnt) - ((pnt_on_line - pnt).dot(line) * line)).norm();
  }

  IModule &LModule::locate_dormer(const DormerModule &dormer) {
    std::vector<Vector2d> footprint1 = this->imodule_first_.footprint_;
    std::vector<Vector2d> footprint2 = this->imodule_second_.footprint_;
    Vector2d dormer_center = dormer.center_.cast<double>();
    //std::cout <<"locate:dom center:" << dormer_center << endl; 
    dormer_center += dormer.offset_;
    if (check_point_in_footprint(footprint1, dormer_center)) {
      //    std::cout << "direct footprint1" << std::endl;
      return this->imodule_first_;
    } else if (check_point_in_footprint(footprint2, dormer_center)) {
      //std::cout << "direct footprint2" << std::endl;
      return this->imodule_second_;
    } else {
      double min_dist_1 = std::numeric_limits<double>::max();
      double min_dist_2 = std::numeric_limits<double>::max();
      std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> footprint1_lines
        { std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint1[1] - footprint1[0], footprint1[0]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint1[3] - footprint1[0], footprint1[0]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint1[2] - footprint1[3], footprint1[3]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint1[1] - footprint1[2], footprint1[2])
	    };
      std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> footprint2_lines
        {std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint2[1] - footprint2[0], footprint2[0]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint2[3] - footprint2[0], footprint2[0]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint2[2] - footprint2[3], footprint2[3]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint2[1] - footprint2[2], footprint2[2])
	    };
      //    std::cout << "center is " << dormer_center[0] <<"," << dormer_center[1] << std::endl;
      for (auto &l: footprint1_lines) {
	double d = compute_distance(dormer_center, l.second, l.first);
	//      std::cout << "distance to " << l.second[0]<< "," << l.second[1] <<" is " << d << std::endl;
	if (d < min_dist_1) {
	  min_dist_1 = d;
	}
      }
      for (auto &l : footprint2_lines) {
	double d = compute_distance(dormer_center, l.second, l.first);
	//      std::cout << "distance to " << l.second[0]<< "," << l.second[1] <<" is " << d << std::endl;
	if (d < min_dist_2) {
	  min_dist_2 = d;
	}
      }
      //    std::cout << "min dist 1: " << min_dist_1 << std::endl;
      //    std::cout << "min dist 2: " << min_dist_2 << std::endl;

      if (min_dist_1 < min_dist_2) {
	//	  std::cout << "min dist 1" << std::endl;
	return this->imodule_first_;
      } else {
	//  std::cout << "min dist 2" << std::endl;
	return this->imodule_second_;
      }
    }
  }
  IModule &UModule::locate_dormer(const DormerModule &dormer) {
    std::vector<Vector2d> footprint1 = this->imodule_first_.footprint_;
    std::vector<Vector2d> footprint2 = this->imodule_second_.footprint_;
    std::vector<Vector2d> footprint3 = this->imodule_third_.footprint_;
    Vector2d dormer_center = dormer.center_.cast<double>();
    dormer_center += dormer.offset_;
    if (check_point_in_footprint(footprint1, dormer_center)) {
      //    std::cout << "direct footprint1" << std::endl;
      return this->imodule_first_;
    } else if (check_point_in_footprint(footprint2, dormer_center)) {
      //    std::cout << "direct footprint2" << std::endl;
      return this->imodule_second_;
    } else if(check_point_in_footprint(footprint3, dormer_center)){
      return this->imodule_third_;
    }else{
    
      std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> footprint1_lines
        { std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint1[1] - footprint1[0], footprint1[0]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint1[3] - footprint1[0], footprint1[0]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint1[2] - footprint1[3], footprint1[3]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint1[1] - footprint1[2], footprint1[2])
	    };
      std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> footprint2_lines
        {std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint2[1] - footprint2[0], footprint2[0]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint2[3] - footprint2[0], footprint2[0]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint2[2] - footprint2[3], footprint2[3]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint2[1] - footprint2[2], footprint2[2])
	    };
      std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> footprint3_lines
	{std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint3[1] - footprint3[0], footprint3[0]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint3[3] - footprint3[0], footprint3[0]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint3[2] - footprint3[3], footprint3[3]),
	    std::pair<Eigen::Vector2d, Eigen::Vector2d>(footprint3[1] - footprint3[2], footprint3[2])
	    };
      double min_dist_1 = std::numeric_limits<double>::max();
      double min_dist_2 = std::numeric_limits<double>::max();
      double min_dist_3 = std::numeric_limits<double>::max();
      //    std::cout << "center is " << dormer_center[0] <<"," << dormer_center[1] << std::endl;
      for (auto &l: footprint1_lines) {
	double d = compute_distance(dormer_center, l.second, l.first);
	//      std::cout << "distance to " << l.second[0]<< "," << l.second[1] <<" is " << d << std::endl;
	if (d < min_dist_1) {
	  min_dist_1 = d;
	}
      }
      for (auto &l : footprint2_lines) {
	double d = compute_distance(dormer_center, l.second, l.first);
	//      std::cout << "distance to " << l.second[0]<< "," << l.second[1] <<" is " << d << std::endl;
	if (d < min_dist_2) {
	  min_dist_2 = d;
	}
      }
      for (auto &l : footprint3_lines) {
	double d = compute_distance(dormer_center, l.second, l.first);
	//      std::cout << "distance to " << l.second[0]<< "," << l.second[1] <<" is " << d << std::endl;
	if (d < min_dist_3) {
	  min_dist_3 = d;
	}
      }
      //    std::cout << "min dist 1: " << min_dist_1 << std::endl;
      //    std::cout << "min dist 2: " << min_dist_2 << std::endl;

      if (min_dist_1 < min_dist_2) {
	//      std::cout << "min dist 1" << std::endl;
	if(min_dist_1 < min_dist_3){
	  return this->imodule_first_;
	}else{
	  return this->imodule_third_;
	}
      } else {
	if(min_dist_2 < min_dist_3){
	  return this->imodule_second_;
	}else{
	  return this->imodule_third_;
	}
      }
    }
  }

  bool IModule::assign_chimneys_new(const std::vector<ChimneyModule> &chimneyss, const std::vector<int>& centers_inds) {
    for(unsigned i = 0; i < chimneyss.size(); ++i)
      chimneys.push_back(chimneyss[i]);
    return true;
  }
  bool LModule::assign_chimneys_new(const std::vector<ChimneyModule> &chimneyss, const std::vector<int>& centers_inds) {
    for(unsigned i = 0; i < chimneyss.size(); ++i)
      chimneys.push_back(chimneyss[i]);
    return true;
  }
  bool UModule::assign_chimneys_new(const std::vector<ChimneyModule> &chimneyss, const std::vector<int>& centers_inds) {
    for(unsigned i = 0; i < chimneyss.size(); ++i)
      chimneys.push_back(chimneyss[i]);
    return true;
  }
  bool TwoIModule::assign_chimneys_new(const std::vector<ChimneyModule> &chimneyss, const std::vector<int>& centers_inds) {
    for(unsigned i = 0; i < chimneyss.size(); ++i) {
      if(centers_inds[i] == 0)
        imodule_first_.chimneys.push_back(chimneyss[i]);
      else
        imodule_second_.chimneys.push_back(chimneyss[i]);
    }
    return true;
  }
  bool PiModule::assign_chimneys_new(const std::vector<ChimneyModule> &chimneyss, const std::vector<int>& centers_inds) {
    for(unsigned i = 0; i < chimneyss.size(); ++i) {
      if(centers_inds[i] == 0){
        imodule_first_.chimneys.push_back(chimneyss[i]);
      }
      else if (centers_inds[i] == 1){
        imodule_second_.chimneys.push_back(chimneyss[i]);
      }
      else {
        imodule_third_.chimneys.push_back(chimneyss[i]);
      }
    }
    return true;
  }

  bool IModule::assign_dormers_new(const std::vector<DormerModule> &dormerss, const std::vector<int>& centers_inds) {
    for(unsigned i = 0; i < dormerss.size(); ++i)
      dormers.push_back(dormerss[i]);
    return true;
  }
  bool TwoIModule::assign_dormers_new(const std::vector<DormerModule> &dormers, const std::vector<int>& centers_inds) {
    for(unsigned i = 0; i < dormers.size(); ++i) {
      if(centers_inds[i] == 0)
        imodule_first_.dormers.push_back(dormers[i]);
      else
        imodule_second_.dormers.push_back(dormers[i]);
    }
    return true;
  }
  bool LModule::assign_dormers_new(const std::vector<DormerModule> &dormers, const std::vector<int>& centers_inds) {
    for (auto &dormer : dormers) {
    // for(unsigned i = 0; i < dormers.size(); ++i) {
    //   if(centers_inds[i] == 0)
    //     imodule_first_.dormers.push_back(dormers[i]);
    //   else
    //     imodule_second_.dormers.push_back(dormers[i]);
    // }      
      locate_dormer(dormer).dormers.push_back(dormer);//TODO: change back
      //    cerr << "assigning dormer:" << 0 << "and result:"
      //         << dormer.center[0] << ", " <<dormer.center[1] << ";"
      //         << dormer.radius[0] << ", " << dormer.radius[1] << ";"
      //         << dormer.triangular_ratio << endl;
      //    std::cout << "assign dormers with type: " << static_cast<int>(dormer.type) << "center: " << dormer.center[0] <<"," << dormer.center[1] << " size: " << dormer.radius << endl;
    }
    return true;
  }
  bool UModule::assign_dormers_new(const std::vector<DormerModule> &dormers, const std::vector<int>& centers_inds) {
    for (auto &dormer : dormers)
      locate_dormer(dormer).dormers.push_back(dormer);//TODO: change back
    return true;
  }
  bool PiModule::assign_dormers_new(const std::vector<DormerModule> &dormers, const std::vector<int>& centers_inds) {
    for(unsigned i = 0; i < dormers.size(); ++i) {
      if(centers_inds[i] == 0){
        imodule_first_.dormers.push_back(dormers[i]);
      }
      else if (centers_inds[i] == 1){
        imodule_second_.dormers.push_back(dormers[i]);
      }
      else {
        imodule_third_.dormers.push_back(dormers[i]);
      }
    }
    return true;
  }
  bool TwoIModule::constructDormerHeatmap(Mat &heatmap) {
    imodule_first_.constructDormerHeatmap(heatmap);
    imodule_second_.constructDormerHeatmap(heatmap);
    return true;
  }

  ChimneyModule::ChimneyModule(const Eigen::Vector2d &center,
           const Eigen::Vector2d & radius,
           const double azimuth,
           const Eigen::Vector2d &orig,
           House &house)
  : center_(center), radius_(radius), azimuth_(azimuth), orig_(orig) {
    RecomputeFootprint(center_, orig_, radius_, azimuth_, footprint_);
    house.generate_owner_and_surface_of_face_helper();
    img_surface = house.img_surface;
  }
  DormerModule::DormerModule(const Eigen::Vector2d &center,
			     const Eigen::Vector2d & radius,
			     const double triangular_ratio,
			     const double azimuth,
			     const Eigen::Vector2d &orig,
			     const Eigen::Vector2d offset,
			     //                           const double ridge_degree,
			     House &house)
    : center_(center), radius_(radius) ,triangular_ratio_(triangular_ratio), azimuth_(azimuth), orig_(orig), offset_(offset), parent(nullptr), initialized(true) {
    RecomputeFootprint(center_, orig_, radius_, azimuth_, footprint_);
    house.generate_owner_and_surface_of_face_helper();
    img_surface = house.img_surface;
    face_owner = house.face_owner;
    house.generate_face_to_same_plane();
    same_plane_map = house.same_plane_map;
    try{
      type_ = check_dormer_type();
    }catch (...){
      type_ = DormerType::INVALID;
      std::fstream fs("invalid.txt");
      fs << "failed once" << endl;
    }
    cv::Mat cm_same_plane;
    //  if(!temp_var){
    //    cv::Mat same_plane_mat = face_owner.clone();
    //    for (int j = 0; j < same_plane_mat.rows; j++) {
    //      for (int i = 0; i < same_plane_mat.cols; i++) {
    //        same_plane_mat.at <unsigned char> (j, i) = same_plane_map[face_owner.at <unsigned char> (j, i)];
    //      }
    //    }
    //    cm_same_plane.create(same_plane_mat.rows, same_plane_mat.cols, CV_8UC3);
    //    same_plane_mat *= 5;
    //    applyColorMap(same_plane_mat, cm_same_plane, cv::COLORMAP_JET);
    //    cv::imwrite(std::string("sameplane.png"), cm_same_plane);
    //    temp_var=true;
    //  }
  }

  bool ChimneyModule::Reconstruct(Mesh &mesh, bool LandU) {
    Mesh mesh_temp;
    std::vector<Vector3d> vertices_temp;
    vertices_temp.clear();
    std::vector<float> v_h;
    if(footprint_.size() == 0){
      cerr << "error: footprint_.size() == 0 " << endl;
      return false;
    }
    for(auto&& vertex : footprint_)
      v_h.push_back(img_surface.at<float>(vertex[1], vertex[0]));
    int add_height_seed = rand() % 3;
    double add_height = static_cast<float>(add_height_seed) + 1.5;
    double base_height = static_cast<float>(std::accumulate(v_h.begin(), v_h.end(), 0)) / v_h.size() + add_height;
    for (auto && v : footprint_)
      vertices_temp.push_back(Vector3d(v[0], v[1], base_height));
    for (auto && v : footprint_)
      vertices_temp.push_back(Vector3d(v[0], v[1], 0.0));
    mesh_temp.vertices = vertices_temp;
    ConstructWallMesh(4, mesh_temp);
    mesh_temp.faces.emplace_back(Vector3i(0, 1, 2));
    mesh_temp.faces.emplace_back(Vector3i(0, 2, 3));
    mesh.Merge(mesh_temp);
  }

  bool DormerModule::Reconstruct(Mesh &mesh, bool LandU) {
    if (parent == nullptr) {
      std::cerr << "dormer module at: " << center_ << " refuse to reconstruct: parent not set" << std::endl;
      return false;
    }
    if(radius_[0] == 0 || radius_[1] == 0){
      std::cout << "dormer module at: " << center_ << " ignored: 0 radius_" << std::endl;
      return false;
    }
    if (type_ == DormerType::INVALID){
         // cerr << "dormer module at: " << center_ << "refuse to construct: invalid parameters" << endl;
      return false;
    }
    //  std::cout << "reconstructing: " << center_ << ", " << radius_ << std::endl;
    //bad hack: remove later
    if (triangular_ratio_ > 0.7){
      triangular_ratio_ = 0.7;
    }
    Mesh mesh_temp;
    std::vector<Vector3d> vertices_temp;
    vertices_temp.clear();

    std::vector<Vector2d> footprint_copy = footprint_;
    for (auto &v : footprint_copy) {
      v += offset_;
    }
    if ((footprint_copy[3] - footprint_copy[0]).norm() > (footprint_copy[1] - footprint_copy[0]).norm())
      std::rotate(footprint_copy.begin(), footprint_copy.begin() + 1, footprint_copy.end());

    std::vector<Vector2d> paren_footprint_copy = parent->footprint_;
    double length1 = (paren_footprint_copy[1] - paren_footprint_copy[0]).norm();
    double length2 = (paren_footprint_copy[3] - paren_footprint_copy[0]).norm();
    Eigen::Vector2d long_axis;

    if (LandU)
      long_axis = paren_footprint_copy[1] - paren_footprint_copy[0];
    else if (length1 > length2)
      long_axis = paren_footprint_copy[1] - paren_footprint_copy[0];
    else
      long_axis = paren_footprint_copy[3] - paren_footprint_copy[0];

    Eigen::Vector2d mid_point1;
    Eigen::Vector2d mid_point2;
    double proj1 = std::abs(
			    (footprint_copy[1] - footprint_copy[0]).dot(long_axis) / (footprint_copy[1] - footprint_copy[0]).norm()
			    / long_axis.norm());
    double proj2 = std::abs((footprint_copy[3] - footprint_copy[0]).dot(long_axis) / (footprint_copy[3] - footprint_copy[0]).norm()
			    / long_axis.norm());
    //  std::cout << "center height" << height << std::endl;
    if (proj1 > proj2) {
      mid_point1 = (footprint_copy[1] + footprint_copy[0]) / 2;
      mid_point2 = (footprint_copy[3] + footprint_copy[2]) / 2;
      Vector2d midpont21 = (mid_point2 - mid_point1);
      // Vector2d truecenter = mid_point1 + triangular_ratio_ * midpont21 ;
      // double bottom_length = ((footprint_copy[3] - footprint_copy[2]).norm() + (footprint_copy[1] - footprint_copy[0]).norm()) / 4;
      //    double dormer_height = bottom_length * std::tan(this->ridge_degree / 180 * PI);
      double dormer_height = std::max(img_surface.at<float>(std::round(mid_point1[1]-offset_[1]), std::round(mid_point1[0]-offset_[0])), img_surface.at<float>(std::round(mid_point2[1]-offset_[1]), std::round(mid_point2[0]-offset_[0])));
      //    std::cout << "dormer height" << dormer_height << std::endl;
      //double base_height =  img_surface.at<float>(round(truecenter[1]), round(truecenter[0]));
      double base_height1 = img_surface.at<float>(round(mid_point1[1]), round(mid_point1[0]));
      double base_height2 = img_surface.at<float>(round(mid_point2[1]), round(mid_point2[0]));
      double min_height = std::min(base_height1, base_height2);
      double max_height = std::max(base_height1, base_height2);
      double base_height = (1 - triangular_ratio_) * min_height + triangular_ratio_ * max_height;
      if(base_height == 0){
	     std::cout << "first dormer module at: " << center_.transpose() << " base height: 0" << std::endl;
	     return false;
      }
      // TODO(H): delete
      // dormer_height = base_height + 0.8;
      for (auto && v : footprint_copy)
        vertices_temp.push_back(Vector3d(v[0], v[1], base_height));
      for (auto && v : footprint_copy)
        vertices_temp.push_back(Vector3d(v[0], v[1], 0.0));

      mesh_temp.vertices = vertices_temp;
      ConstructWallMesh(4, mesh_temp);
      mesh_temp.vertices.push_back(Eigen::Vector3d(mid_point1[0],
						   mid_point1[1],
						   dormer_height));//TODO: change to some height
      mesh_temp.vertices.push_back(Eigen::Vector3d(mid_point2[0],
						   mid_point2[1],
						   dormer_height));//TODO: change to some height
      mesh_temp.faces.push_back(Vector3i(0, 1, 8));
      mesh_temp.faces.push_back(Vector3i(2, 3, 9));
      mesh_temp.faces.push_back(Vector3i(3, 8, 9));
      mesh_temp.faces.push_back(Vector3i(3, 0, 8));
      mesh_temp.faces.push_back(Vector3i(2, 9, 8));
      mesh_temp.faces.push_back(Vector3i(2, 1, 8));

    } else {
      mid_point1 = (footprint_copy[3] + footprint_copy[0]) / 2;
      mid_point2 = (footprint_copy[2] + footprint_copy[1]) / 2;
      Vector2d midpont21 = (mid_point2 - mid_point1);
      Vector2d truecenter = mid_point1 + triangular_ratio_ * midpont21 ;
      double bottom_length = ((footprint_copy[3] - footprint_copy[0]).norm() + (footprint_copy[2] - footprint_copy[1]).norm()) / 4;
      //    double dormer_height = bottom_length * std::tan(this->ridge_degree / 180 * PI);
      double dormer_height = std::max(img_surface.at<float>(std::round(mid_point1[1]-offset_[1]), std::round(mid_point1[0]-offset_[0])), img_surface.at<float>(std::round(mid_point2[1]-offset_[1]), std::round(mid_point2[0]-offset_[0])));
      //    std::cout << "dormer height" << dormer_height << std::endl;
      double base_height = img_surface.at<float>(round(truecenter[1]), round(truecenter[0]));
      // if(base_height == 0){
	     // std::cout << "dormer module at: " << center.transpose() << " base height: 0" << std::endl;
      // 	return false;
      // }

      // TODO(H): delete
      // dormer_height = base_height + 0.8;

      for (auto && v : footprint_copy)
	     vertices_temp.push_back(Vector3d(v[0], v[1], base_height));

      for (auto && v : footprint_copy)
	     vertices_temp.push_back(Vector3d(v[0], v[1], 0.0));

      mesh_temp.vertices = vertices_temp;
      ConstructWallMesh(4, mesh_temp);
      mesh_temp.vertices.push_back(Eigen::Vector3d(mid_point1[0],
						   mid_point1[1],
						   dormer_height));//TODO: change to some height
      mesh_temp.vertices.push_back(Eigen::Vector3d(mid_point2[0],
						   mid_point2[1],
						   dormer_height));//TODO: change to some height
      mesh_temp.faces.push_back(Vector3i(0, 3, 8));
      mesh_temp.faces.push_back(Vector3i(1, 2, 9));
      mesh_temp.faces.push_back(Vector3i(0, 1, 8));
      mesh_temp.faces.push_back(Vector3i(1, 8, 9));
      mesh_temp.faces.push_back(Vector3i(2, 8, 9));
      mesh_temp.faces.push_back(Vector3i(2, 3, 8));
    }
    mesh.Merge(mesh_temp);
    return true;
  }
  bool DormerModule::constructDormerHeatmap(Mat &heatmap) {
    // std::cout << center_[0] <<", " << center_[1] << std::endl;
    heatmap.at<unsigned char>(center_[1], center_[0]) = 255;
    return true;
  }
  DormerModule::DormerModule(): initialized(false) {

  }

  DormerType DormerModule::check_dormer_type() {
    assert(false);
    // unsigned char curr_owner = face_owner.at < unsigned char > (center_[1], center_[0]);
    // unsigned char mapped_owner = same_plane_map.at(curr_owner);
    // //        validity.create(face_owner.rows, face_owner.cols, face_owner.type());
    // //        validity.setTo(cv::Scalar(0));
    // double azimuth_degree = azimuth_ * 180 / PI;
    // Eigen::Vector2d orig(face_owner.cols / 2, face_owner.rows / 2);
    // Eigen::Vector2d rotated_center = rotate_pnt(center_.cast<double>(), orig, -azimuth_degree);
    // double x_min = rotated_center[0] - radius_[0];
    // double x_max = rotated_center[0] + radius_[0];
    // double y_min = rotated_center[1] - radius_[1];
    // double y_max = rotated_center[1] + radius_[1];
    DormerType type = DormerType::DORMER;
 //    for (int j = std::round(y_min); j <= std::round(y_max); j++) {
 //      for (int i = std::round(x_min); i <= std::round(x_max); i++) {
	// Eigen::Vector2d rotated_point = rotate_pnt(Eigen::Vector2d(i, j), orig, azimuth_degree);
	// Eigen::Vector2d rotated_round_point = ArrayXd::round(rotated_point.array()).matrix();
	// //                    validity.at<unsigned char>(rotated_round_point[1], rotated_round_point[0]) = 100;
	// if (rotated_round_point[0] < 0 || rotated_round_point[0] >= face_owner.cols || rotated_round_point[1] < 0
	//     || rotated_round_point[1] >= face_owner.rows) {
	//   continue;
	// }
	// unsigned char new_mapped_owner = same_plane_map.at(face_owner.at <unsigned char> (rotated_round_point[1], rotated_round_point[0]));
	// if (new_mapped_owner != mapped_owner){
	//   if(face_owner.at <unsigned char> (rotated_round_point[1], rotated_round_point[0]) == 0){
	//     if(type == DormerType::DORMER){
	//       type = DormerType::ATTACHMENT;
	//     }
	//   }else{
	//     if( center_[0]==22 && center_[1] == 33 && radius_[0] == 3 && radius_[1] == 3){
	//       cerr << "invalid due to point at" <<rotated_round_point[0]  << "," <<rotated_round_point[1]<< "owner:"
	// 	   <<(int)face_owner.at <unsigned char> (rotated_round_point[1], rotated_round_point[0])
	// 	   << endl;
	//     }

	//     type = DormerType::INVALID;
	//   }
	// }
	//                    validity.at<unsigned char>(rotated_round_point[1], rotated_round_point[0]) = 255;
    //   }
    // }
    return type;
  }

  std::vector<std::vector<Eigen::Vector2d>> Cad::get_footprints() const {
    std::vector<std::vector<Eigen::Vector2d>> result;
    if(p_module_root_->get_module_type() == ModuleType::kPiModule) {
      auto p_three_i = std::dynamic_pointer_cast<PiModule>(p_module_root_);
      result.push_back(p_three_i.get()->imodule_first_.footprint_);  
      result.push_back(p_three_i.get()->imodule_second_.footprint_); 
      result.push_back(p_three_i.get()->imodule_third_.footprint_);
    }
    if(p_module_root_->get_module_type() == ModuleType::kTwoIModule) {
      auto p_two_i = std::dynamic_pointer_cast<TwoIModule>(p_module_root_);
      result.push_back(p_two_i.get()->imodule_first_.footprint_);  
      result.push_back(p_two_i.get()->imodule_second_.footprint_); 
    }
    return result;
  }

  std::vector<Eigen::Vector2d> Cad::get_dormers() const {
    std::vector<Eigen::Vector2d> result;
    if(p_module_root_->get_module_type() == ModuleType::kPiModule) {
      auto p_three_i = std::dynamic_pointer_cast<PiModule>(p_module_root_);
      for(auto&& dormer : p_three_i.get()->imodule_first_.dormers)
        result.push_back(dormer.center_);
      for(auto&& dormer : p_three_i.get()->imodule_second_.dormers)
        result.push_back(dormer.center_);
      for(auto&& dormer : p_three_i.get()->imodule_third_.dormers)
        result.push_back(dormer.center_);
    }
    if(p_module_root_->get_module_type() == ModuleType::kTwoIModule) {
      auto p_two_i = std::dynamic_pointer_cast<TwoIModule>(p_module_root_);
      for(auto&& dormer : p_two_i.get()->imodule_first_.dormers)
        result.push_back(dormer.center_);
      for(auto&& dormer : p_two_i.get()->imodule_second_.dormers)
        result.push_back(dormer.center_);
    }
    return result;
  }

  std::vector<Eigen::Vector2d> Cad::get_chimneys() const {
    std::vector<Eigen::Vector2d> result;
    if(p_module_root_->get_module_type() == ModuleType::kPiModule) {
      auto p_three_i = std::dynamic_pointer_cast<PiModule>(p_module_root_);
      for(auto&& chimney : p_three_i.get()->imodule_first_.chimneys)
        result.push_back(chimney.center_);
      for(auto&& chimney : p_three_i.get()->imodule_second_.chimneys)
        result.push_back(chimney.center_);
      for(auto&& chimney : p_three_i.get()->imodule_third_.chimneys)
        result.push_back(chimney.center_);
    }
    if(p_module_root_->get_module_type() == ModuleType::kTwoIModule) {
      auto p_two_i = std::dynamic_pointer_cast<TwoIModule>(p_module_root_);
      for(auto&& chimney : p_two_i.get()->imodule_first_.chimneys)
        result.push_back(chimney.center_);
      for(auto&& chimney : p_two_i.get()->imodule_second_.chimneys)
        result.push_back(chimney.center_);
    }
    return result;
  }
  double Cad::get_azimuth() const {
    return static_cast<double>(p_module_root_->get_azimuth());
  }


  bool Cad::AddOverhang() {
    assert(p_module_root_->get_module_type() == ModuleType::kTwoIModule && "not two-i module, not processing");
    auto p_two_i = std::dynamic_pointer_cast<TwoIModule>(p_module_root_);
    std::vector<RoofType> rooftypes{p_two_i.get()->imodule_first_.get_rooftype(), p_two_i.get()->imodule_second_.get_rooftype()};

    // assert(p_module_root_->get_module_type() == ModuleType::kPiModule && "not 3-i module, not processing");
    // auto p_three_i = std::dynamic_pointer_cast<PiModule>(p_module_root_);
    // std::vector<RoofType> rooftypes{p_three_i.get()->imodule_first_.get_rooftype(), p_three_i.get()->imodule_second_.get_rooftype(), p_three_i.get()->imodule_third_.get_rooftype()};

    // assert(p_module_root_->get_module_type() == ModuleType::kLModule && "not L module, not processing");
    // auto p_l = std::dynamic_pointer_cast<LModule>(p_module_root_);
    // std::vector<RoofType> rooftypes{p_l.get()->imodule_first_.get_rooftype(), p_l.get()->imodule_second_.get_rooftype()};

    cerr << "roof type" << endl;
    cerr << static_cast<std::underlying_type<RoofType>::type>(rooftypes[0]) << endl;
    cerr << static_cast<std::underlying_type<RoofType>::type>(rooftypes[1]) << endl;
    // for(unsigned ind_i_module = 0; ind_i_module < 3; ++ind_i_module) {
    for(unsigned ind_i_module = 0; ind_i_module < 2; ++ind_i_module) {
        Mesh mesh_o; // overhang
        mesh_o.vertices.resize(12); // this applies for all roof type
        cerr << "mesh_.vertices.size(): " << mesh_.vertices.size() << endl;
        std::vector<Vector3d> vertices(mesh_.vertices.begin() + ind_i_module * 10, mesh_.vertices.begin() + ind_i_module * 10 + 10);
        if (rooftypes[ind_i_module] == RoofType::kHip) {
        	mesh_o.vertices[8] = vertices[8];
        	mesh_o.vertices[9] = vertices[9];
        	auto v00 = (vertices[0] - mesh_o.vertices[8]) * 1.15 + mesh_o.vertices[8]; auto v11 = (vertices[1] - mesh_o.vertices[9]) * 1.15 + mesh_o.vertices[9];
        	auto v22 = (vertices[2] - mesh_o.vertices[9]) * 1.15 + mesh_o.vertices[9]; auto v33 = (vertices[3] - mesh_o.vertices[8]) * 1.15 + mesh_o.vertices[8];
        	mesh_o.vertices[0] = v00; mesh_o.vertices[1] = v11; mesh_o.vertices[2] = v22; mesh_o.vertices[3] = v33;
        	Vector3d thick = Vector3d(0, 0, 0.3);
        	for(unsigned i = 0; i < 4; ++i)
        	  mesh_o.vertices[i + 4] = mesh_o.vertices[i] + thick;
        	mesh_o.vertices[10] = mesh_o.vertices[8] + thick; mesh_o.vertices[11] = mesh_o.vertices[9] + thick;
        	mesh_o.faces.push_back(Vector3i(0, 1, 8));
        	mesh_o.faces.push_back(Vector3i(1, 8, 9));
        	mesh_o.faces.push_back(Vector3i(1, 2, 9));
        	mesh_o.faces.push_back(Vector3i(2, 3, 9));
        	mesh_o.faces.push_back(Vector3i(3, 8, 9));
        	mesh_o.faces.push_back(Vector3i(0, 3, 8));

          mesh_o.faces.push_back(Vector3i(4, 5, 10));
          mesh_o.faces.push_back(Vector3i(5, 10, 11));
          mesh_o.faces.push_back(Vector3i(5, 6, 11));
          mesh_o.faces.push_back(Vector3i(6, 7, 11));
          mesh_o.faces.push_back(Vector3i(7, 10, 11));
          mesh_o.faces.push_back(Vector3i(4, 7, 10));
          ConstructWallMesh(4, mesh_o);
      }
      else { // Gable
        cerr << "mesh_o.vertices.size(): " << mesh_o.vertices.size() << endl;

        // I shape
        // mesh_o.vertices[8] = (vertices[8] - vertices[9]) * 1.05 + vertices[9]; 
        // mesh_o.vertices[0] = (vertices[0] - vertices[1]) * 1.05 + vertices[1]; 
        // mesh_o.vertices[3] = (vertices[3] - vertices[2]) * 1.05 + vertices[2]; 
        // L shape
        mesh_o.vertices[8] = (vertices[8] - vertices[9]) * 1.00 + vertices[9]; 
        mesh_o.vertices[0] = (vertices[0] - vertices[1]) * 1.00 + vertices[1]; 
        mesh_o.vertices[3] = (vertices[3] - vertices[2]) * 1.00 + vertices[2]; 

        mesh_o.vertices[9] = (vertices[9] - vertices[8]) * 1.05 + vertices[8];
        mesh_o.vertices[1] = (vertices[1] - vertices[0]) * 1.05 + vertices[0];
        mesh_o.vertices[2] = (vertices[2] - vertices[3]) * 1.05 + vertices[3];
        auto v00 = (mesh_o.vertices[0] - mesh_o.vertices[8]) * 1.15 + mesh_o.vertices[8]; auto v11 = (mesh_o.vertices[1] - mesh_o.vertices[9]) * 1.15 + mesh_o.vertices[9];
      	auto v22 = (mesh_o.vertices[2] - mesh_o.vertices[9]) * 1.15 + mesh_o.vertices[9]; auto v33 = (mesh_o.vertices[3] - mesh_o.vertices[8]) * 1.15 + mesh_o.vertices[8];
      	mesh_o.vertices[0] = v00; mesh_o.vertices[1] = v11; mesh_o.vertices[2] = v22; mesh_o.vertices[3] = v33;
      	Vector3d thick = Vector3d(0, 0, 0.35);
        for(unsigned i = 0; i < 4; ++i)
      	  mesh_o.vertices[i + 4] = mesh_o.vertices[i] + thick;
      	mesh_o.vertices[10] = mesh_o.vertices[8] + thick; mesh_o.vertices[11] = mesh_o.vertices[9] + thick;
      	mesh_o.faces.push_back(Vector3i(0, 1, 8));
      	mesh_o.faces.push_back(Vector3i(1, 8, 9));
      	mesh_o.faces.push_back(Vector3i(2, 3, 9));
      	mesh_o.faces.push_back(Vector3i(3, 8, 9));
        mesh_o.faces.push_back(Vector3i(0+4, 1+4, 10));
        mesh_o.faces.push_back(Vector3i(1+4, 10, 11));
        mesh_o.faces.push_back(Vector3i(2+4, 3+4,11));
        mesh_o.faces.push_back(Vector3i(3+4, 10, 11));        
        mesh_o.faces.push_back(Vector3i(0, 1, 4)); mesh_o.faces.push_back(Vector3i(1, 4, 5));
        mesh_o.faces.push_back(Vector3i(2, 3, 6)); mesh_o.faces.push_back(Vector3i(3, 6, 7));
        mesh_o.faces.push_back(Vector3i(1, 5, 11)); mesh_o.faces.push_back(Vector3i(1, 9, 11));
        mesh_o.faces.push_back(Vector3i(2, 6, 11)); mesh_o.faces.push_back(Vector3i(2, 9, 11));
        mesh_o.faces.push_back(Vector3i(1-1, 5-1, 11-1)); mesh_o.faces.push_back(Vector3i(1-1, 9-1, 11-1));
        mesh_o.faces.push_back(Vector3i(3,8,10)); mesh_o.faces.push_back(Vector3i(3,7,10));
      }
      mesh_.Merge(mesh_o);
    }

    // cerr << mesh_ << endl;
    return true;
  }

} // DPM
  
