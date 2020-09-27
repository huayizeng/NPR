#ifndef BASE_HOUSE_H_
#define BASE_HOUSE_H_

#include <memory>
#include <vector>
#include <string>
#include <iostream>


#ifdef GPU_FOUND
#include "rendering_offscreen.h"
#endif

#include "features_dnn.pb.h"
#include <opencv2/core.hpp>
namespace DPM {

// exist because c++11 lacks make_unique
// template<typename T, typename... Args>
// std::unique_ptr<T> make_unique(Args&&... args)
// {
//   return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
// }

class FileIO;
class Lidar;
class Cad;

class House {
private:
  // TODO(Henry): instance cad to be added
  std::shared_ptr<Lidar> p_lidar_;
  std::shared_ptr<Cad> p_cad_;
  std::string name_;
  FeatureDNNProto feature_dnn_;

public:
  House(/* args */);
  House(const House &house);
  ~House();
  House(House &&house);
  House & operator=(const House &) = default;
  bool ComputeDistance(const std::vector<int> &x_request,
                       const std::vector<int> &y_request,
                       std::vector<double> &distance_at_request) const;
  // double ComputeDistanceInOne(const Mat& img_gradient) const;
  #ifdef GPU_FOUND
  bool ComputeDistanceGPU(const std::vector<int>& x_request, const std::vector<int>& y_request,
                                 std::vector<double>& distance_at_request, Corender& mycorender) const;
  #endif

  inline const Lidar &get_lidar() const { return *p_lidar_; }
  inline const Cad &get_cad() const { return *p_cad_; }
  inline const std::string &get_name() const { return name_; }
  inline const FeatureDNNProto &get_feature_dnn() const {return feature_dnn_;}

  void set_lidar(const Lidar &lidar);
  void set_name(const std::string &name_str) { name_ = name_str; }
  void set_cad(const Cad &cad);
  void set_feature_dnn(const FeatureDNNProto &feature_dnn) {feature_dnn_ = feature_dnn;}
  void write_state(const std::string& path);
  bool generate_owner_and_surface_of_face_helper();
  bool ComputeImgSurface(cv::Mat &img_surface) const;
  bool generate_face_to_same_plane();
  cv::Mat img_surface;
  cv::Mat face_owner;
  std::map<unsigned char, unsigned char> same_plane_map;
};

class HouseGroup {
private:
public:
  std::vector<House> houses;
  HouseGroup(/* args */) = default;
};

// For process_real
bool Preprocess(FileIO& file_io, HouseGroup& housegroup);
bool ComputeCad(FileIO& file_io, HouseGroup& housegroup);
void WriteLidarImg(const std::string& dir_depth_img, const std::string& dir_normal_img, HouseGroup & housegroup);
void WriteXYZN(const std::string& dir_xyzn, const HouseGroup & housegroup);
void WritePointCloud(FileIO & file_io, HouseGroup & housegroup);
void WriteCad(const std::string& dir_cad, HouseGroup & housegroup);
void WriteStates(const std::string& dir_states, HouseGroup & housegroup);
void WritePolygonModel(const std::string& dir_polygon_model, HouseGroup & housegroup);
void AddOverhangs(HouseGroup & housegroup);
void ReadHouseFromSavedDir(FileIO & file_io, HouseGroup & housegroup);
bool GetDNNPrediction(FileIO& file_io, HouseGroup& housegroup);
// For process_synthetic
void WriteLatLng(HouseGroup & housegroup, FileIO & file_io);

#ifdef GPU_FOUND
// void RenderForMapMode();
void RenderForDebugFromOptCadDir(FileIO & file_io);
void RenderForDebug(FileIO & file_io, HouseGroup & housegroup);
void RenderForSketchUp(FileIO & file_io);
#endif

} // DPM

#endif  // BASE_HOUSE_H_
