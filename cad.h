#ifndef BASE_CAD_H_
#define BASE_CAD_H_

#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <Eigen/Eigenvalues>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "mesh.h"
#include "file_io.h"
#include "features_dnn.pb.h"
#include "house.h"
namespace DPM {

/*    kI            = 3;
    kII           = 1;
    kIII          = 4;
    kL            = 5;
    kU            = 6;
    kComp         = 2;
*/
enum class ModuleType {
  kBaseModule = 0,
  kIModule = 3,
  kLModule = 5,
  kTwoIModule = 1,
  kDormerModule = 7,
  kUModule = 6,
  kPiModule = 4,
  kCompModule = 2
};

enum class RoofType{
  kGable = 1,
  kHip = 2,
  kFlat = 3
};

enum struct DormerType{
  DORMER = 0,
  ATTACHMENT,
  INVALID
};
class FileIO;
class IModule;

typedef struct OptionRecognition {
  OptionRecognition(const cv::Mat &img_depth_in, const std::string name_house_in, const FileIO &file_io, const FeatureDNNProto& feature_dnn_in)
      : img_depth(img_depth_in),
        name_house(name_house_in),
        p_file_io(std::make_shared<FileIO>(file_io)),
        feature_dnn(feature_dnn_in) {}
  cv::Mat img_depth;
  std::string name_house;
  std::shared_ptr<FileIO> p_file_io;
  FeatureDNNProto feature_dnn;
} OptRecog;
 
class DormerModule;
class ChimneyModule;
class BaseModule {
 public:
  virtual ~BaseModule() = default;
  virtual inline const ModuleType get_module_type() const { return ModuleType::kBaseModule; }
  virtual inline const float get_azimuth() const { return 0; }
  virtual bool ModuleRecognition(const OptRecog &opt_recog) { return false; }
  virtual bool Reconstruct(Mesh &mesh, bool UandI) { return false; }
  // virtual bool Parsing(const Mesh &mesh) { return false; }
  virtual bool assign_dormers(const std::vector<DormerModule> &dormers){return false; };
  virtual bool assign_dormers_new(const std::vector<DormerModule> &dormers, const std::vector<int>& centers_inds) {return false; };
  virtual bool assign_chimneys_new(const std::vector<ChimneyModule> &chimneys, const std::vector<int>& centers_inds) {return false; };
  virtual double calc_azimuth() {return 0/0; };
  virtual bool Reconstruct(Mesh &mesh){
    switch(this->get_module_type()){
    case ModuleType::kIModule:
      return this->Reconstruct(mesh, false);
      break;
    case ModuleType::kLModule:
      return this->Reconstruct(mesh, true);
      break;
    case ModuleType::kTwoIModule:
      return this->Reconstruct(mesh, false);
      break;
    case ModuleType::kDormerModule:
      return this->Reconstruct(mesh, false);
      break;
    case ModuleType::kUModule:
      return this->Reconstruct(mesh, true);
      break;
    case ModuleType::kPiModule:
      return this->Reconstruct(mesh, false);
      break;
    case ModuleType::kCompModule:
      return this->Reconstruct(mesh, false);
      break;
    }
  };
  virtual std::shared_ptr<BaseModule> clone() {
    return std::make_shared<BaseModule>(*this);
  }
};

class ChimneyModule: public BaseModule {
public:
  ChimneyModule(const Eigen::Vector2d &center,
                const Eigen::Vector2d &radius,
                const double azimuth,
                const Eigen::Vector2d &orig,
                House &house); // TODO(H): why there is a house??
  virtual bool Reconstruct(Mesh &mesh, bool UandI) override;
  std::vector<Eigen::Vector2d> footprint_;
  Eigen::Vector2d center_;
  Eigen::Vector2d radius_;
  Eigen::Vector2d orig_;
  cv::Mat img_surface;
  double azimuth_;
  bool initialized;
};

class DormerModule : public BaseModule {
 public:
  DormerModule();
  DormerModule(const Eigen::Vector2d &center,
               const Eigen::Vector2d & radius,
               const double triangular_ratio,
               const double azimuth,
               const Eigen::Vector2d &orig,
               const Eigen::Vector2d offset,
//                           const double ridge_degree,
               House &house);
  virtual bool Reconstruct(Mesh &mesh, bool UandI) override;
  virtual inline const ModuleType get_module_type() const override { return ModuleType::kDormerModule; }

  DormerType check_dormer_type(); //note: not the same as synthetic
  virtual bool ModuleRecognition(const OptRecog &opt_recog);
  bool constructDormerHeatmap(cv::Mat &heatmap); // used in synthesis
  std::vector<Eigen::Vector2d> footprint_;
  Eigen::Vector2d center_;
  Eigen::Vector2d radius_;
  double triangular_ratio_;
  double azimuth_;
  bool initialized;
  Eigen::Vector2d orig_;
//  double ridge_degree;
  cv::Mat img_surface;
  cv::Mat face_owner;
  std::map<unsigned char, unsigned char> same_plane_map;
  DormerType type_;
  Eigen::Vector2d offset_;
  IModule *parent;

  void set_paren(IModule *paren) {
    this->parent = paren;
  }
  virtual std::shared_ptr<BaseModule> clone() override {
    return std::make_shared<DormerModule>(*this);
  }
};
class IFlat : public BaseModule{
public:
  float azimuth_;
  float height_;
  std::vector<Eigen::Vector2d> footprint_;
  void set_footprint(const std::vector<Eigen::Vector2d> &footprint) { footprint_ = footprint; }
  virtual bool Reconstruct(Mesh &mesh, bool UandI) override;
  // virtual bool ModuleRecognition(const OptRecog &opt_recog) override; // get the height_
  virtual std::shared_ptr<BaseModule> clone() override {
    return std::make_shared<IFlat>(*this);
  }
};

class LFlat : public BaseModule{
public:
  float azimuth_;
  float height_;
  std::vector<Eigen::Vector2d> footprint_;
  void set_footprint(const std::vector<Eigen::Vector2d> &footprint) { footprint_ = footprint; }
  virtual bool Reconstruct(Mesh &mesh, bool UandI) override;
  // virtual bool ModuleRecognition(const OptRecog &opt_recog) override; // get the height_
  virtual std::shared_ptr<BaseModule> clone() override {
    return std::make_shared<LFlat>(*this);
  }
};

class IModule : public BaseModule {
 public:
  virtual bool ModuleRecognition(const OptRecog &opt_recog) override;
  virtual bool Reconstruct(Mesh &mesh, bool LandU) override;
  // virtual bool Parsing(const Mesh &mesh) override;
  virtual inline const ModuleType get_module_type() const override { return ModuleType::kIModule; }
  inline const RoofType get_rooftype() const {return rooftype_;}
  inline const float get_height_eave() const { return height_eave_; }
  inline const float get_height_ridge() const { return height_ridge_; }
  inline const float get_ratio_pos_left_end_ridge() const { return ratio_pos_left_end_ridge_; }
  inline const float get_ratio_pos_right_end_ridge() const { return ratio_pos_right_end_ridge_; }
  virtual inline const float get_azimuth() const override { return azimuth_; }
  bool constructDormerHeatmap(cv::Mat &heatmap);
  void set_footprint(const std::vector<Eigen::Vector2d> &footprint) { footprint_ = footprint; }
  virtual bool assign_dormers_new(const std::vector<DormerModule> &dormers, const std::vector<int>& centers_inds) override;
  virtual bool assign_chimneys_new(const std::vector<ChimneyModule> &chimneys, const std::vector<int>& centers_inds) override;
  float azimuth_;
  float height_eave_;
  float height_ridge_;
  float ratio_pos_left_end_ridge_;
  float ratio_pos_right_end_ridge_;
  std::vector<Eigen::Vector2d> footprint_;
  std::vector<DormerModule> dormers;
  std::vector<ChimneyModule> chimneys;
  std::vector<IFlat> iflats_;
  std::vector<LFlat> lflats_;
  RoofType rooftype_;
  bool isIndependent_ = true;
  virtual double calc_azimuth() override;
  virtual std::shared_ptr<BaseModule> clone() override {
    return std::make_shared<IModule>(*this);
  }
};

class LModule : public BaseModule {
 public:
  virtual inline const ModuleType get_module_type() const override { return ModuleType::kLModule; }
  virtual bool ModuleRecognition(const OptRecog &opt_recog) override;
  virtual bool Reconstruct(Mesh &mesh, bool LandU) override;
  bool AdjustingEngPoints(Mesh &mesh);
  float azimuth_;
  IModule imodule_first_;
  IModule imodule_second_;
  std::vector<IFlat> iflats_;
  std::vector<LFlat> lflats_;
  std::vector<ChimneyModule> chimneys;
  virtual bool assign_dormers_new(const std::vector<DormerModule> &dormers, const std::vector<int>& centers_inds) override;
  virtual bool assign_chimneys_new(const std::vector<ChimneyModule> &chimneys, const std::vector<int>& centers_inds) override;
  IModule &locate_dormer(const DormerModule &dormer);
  virtual std::shared_ptr<BaseModule> clone() override {
    return std::make_shared<LModule>(*this);
  }
   virtual double calc_azimuth() override;
   virtual inline const float get_azimuth() const override { return azimuth_; }
};

class TwoIModule : public BaseModule {
 public:
  virtual inline const ModuleType get_module_type() const override { return ModuleType::kTwoIModule; }
  virtual bool ModuleRecognition(const OptRecog &opt_recog) override;
  virtual bool Reconstruct(Mesh &mesh, bool LandU) override;
  // virtual bool Parsing(const Mesh &mesh) override;
  virtual double calc_azimuth() override;
  virtual inline const float get_azimuth() const override { return azimuth_; }
  bool constructDormerHeatmap(cv::Mat &heatmap);
  IModule &locate_dormer(const DormerModule &dormer);
  float azimuth_;
  // std::vector<Eigen::Vector2d> footprint_i_first_;
  // std::vector<Eigen::Vector2d> footprint_i_second_;
  IModule imodule_first_;
  IModule imodule_second_;
  std::vector<IFlat> iflats_;
  std::vector<LFlat> lflats_;

  virtual std::shared_ptr<BaseModule> clone() override {
    return std::make_shared<TwoIModule>(*this);
  } 
  virtual bool assign_dormers_new(const std::vector<DormerModule> &dormers, const std::vector<int>& centers_inds) override;
  virtual bool assign_chimneys_new(const std::vector<ChimneyModule> &chimneys, const std::vector<int>& centers_inds) override;
};


class UModule : public BaseModule {
 public:
  virtual inline const ModuleType get_module_type() const override { return ModuleType::kUModule; }
  virtual bool ModuleRecognition(const OptRecog &opt_recog) override;
  virtual bool Reconstruct(Mesh &mesh, bool LandU) override;
  bool AdjustingEngPointsU(  int ind_inner_end1, int ind_inner_end12,
  int ind_inner_end2,  int ind_inner_end22, 
  int ind_helper, 
  int ind_helper2, int ind_helper3, Mesh &mesh);
  float azimuth_;
  IModule imodule_first_;
  IModule imodule_second_;
  IModule imodule_third_;
  std::vector<IFlat> iflats_;
  std::vector<LFlat> lflats_;
  std::vector<ChimneyModule> chimneys;
  IModule &locate_dormer(const DormerModule &dormer);
  virtual std::shared_ptr<BaseModule> clone() override {
    return std::make_shared<UModule>(*this);
  }
  virtual bool assign_dormers_new(const std::vector<DormerModule> &dormers, const std::vector<int>& centers_inds) override;
  virtual bool assign_chimneys_new(const std::vector<ChimneyModule> &chimneys, const std::vector<int>& centers_inds) override;
  virtual double calc_azimuth() override;
  virtual inline const float get_azimuth() const override { return azimuth_; }
};

class PiModule : public BaseModule {
 public:
  virtual inline const ModuleType get_module_type() const override { return ModuleType::kPiModule; }
  virtual bool ModuleRecognition(const OptRecog &opt_recog) override;
  virtual bool Reconstruct(Mesh &mesh, bool LandU) override;
  // virtual bool Parsing(const Mesh &mesh) override;
  float azimuth_;
  IModule imodule_first_;
  IModule imodule_second_;
  IModule imodule_third_;
  std::vector<IFlat> iflats_;
  std::vector<LFlat> lflats_;

  IModule &locate_dormer(const DormerModule &dormer);
  virtual std::shared_ptr<BaseModule> clone() override {
    return std::make_shared<PiModule>(*this);
  } 
  virtual bool assign_dormers_new(const std::vector<DormerModule> &dormers, const std::vector<int>& centers_inds) override;
  virtual bool assign_chimneys_new(const std::vector<ChimneyModule> &chimneys, const std::vector<int>& centers_inds) override;
  virtual double calc_azimuth() override;
  virtual inline const float get_azimuth() const override { return azimuth_; }
};

class CompModule: public BaseModule {
  public:
    virtual inline const ModuleType get_module_type() const override {return ModuleType::kCompModule;}
    // virtual bool ModuleRecognition(const OptRecog &opt_recog) override;
    // virtual bool Reconstruct(Mesh &mesh) override;
    // virtual bool Parsing(const Mesh &mesh) override;
    float azimuth_;
    std::vector<std::vector<Eigen::Vector2d> >  footprints_;
    std::vector<IModule> imodules_;
    virtual std::shared_ptr<BaseModule> clone() override {
      return std::make_shared<CompModule>(*this);
    }
};

class Cad {
 private:
  bool is_reconstructed_;
  Mesh mesh_;
 public:
  // TODO(Henry): we should put this back to private someday
  std::shared_ptr<BaseModule> p_module_root_;
  Cad()
      : is_reconstructed_(false),
        p_module_root_(std::make_shared<BaseModule>()) {}
  bool Recognition(const OptRecog &opt_recog);
  bool ReconstuctAll(bool LandU) {
    mesh_.vertices.clear();
    mesh_.faces.clear();
    bool result = p_module_root_->Reconstruct(mesh_, LandU);
    is_reconstructed_ = true;
    return result;
  }
  bool ReconstuctAll() {
    mesh_.vertices.clear();
    mesh_.faces.clear();
    bool result = p_module_root_->Reconstruct(mesh_);
    is_reconstructed_ = true;
    return result;
  }
  // void WriteMesh(const std::string &filename) const { mesh_.Write(filename); }
  void WriteMesh(const std::string &filename) { 
    for(auto&& vertex : mesh_.vertices)
      vertex[0] = vertex[0] * -1;
    mesh_.Write(filename); 
  }
  // bool Subsampling(cv::Mat &img_depth, Eigen::Vector2d& offset);
  // bool Parsing();
  bool AddOverhang();
  std::vector<std::vector<Eigen::Vector2d>> get_footprints() const;
  std::vector<Eigen::Vector2d> get_dormers() const;
  std::vector<Eigen::Vector2d> get_chimneys() const;
  double get_azimuth() const;
  // Accessor
  inline const ModuleType get_module_type() const { return p_module_root_->get_module_type(); }
  inline const Mesh &get_mesh() const { return mesh_; }
  inline const bool is_reconstructed() const { return is_reconstructed_; }
  inline void set_mesh(const Mesh &mesh) { mesh_ = mesh; }
  inline void set_is_reconstructed(const bool &is_reconstructed) { is_reconstructed_ = is_reconstructed; }

};

} // DPM

#endif  // BASE_CAD_H_

