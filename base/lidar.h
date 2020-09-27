#ifndef BASE_LIDAR_H_
#define BASE_LIDAR_H_

#include <memory>
#include <vector>

#include <Eigen/Eigenvalues>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"


namespace DPM {

class Lidar {

public:
  Lidar() = default;
  Lidar(const Lidar& lidar)
    : img_depth_(lidar.img_depth_.clone()),
      offset_(lidar.offset_) {}

  bool ReadImgDepth(const std::string& filename_dsm, const std::string& filename_dtm);
  bool ReadImgDepth(const std::string& filename_diff);
  bool ReadImgDepthObj(const std::string& filename);
  inline void RemoveLowerPoints(const int kLowestHeight) {
    img_depth_.setTo(0, img_depth_ < kLowestHeight);
  }
  bool AddNoise();
  bool KeepCenterConnectedComponent();
  void GetColoredDepthImage(cv::Mat& img_colored);
  bool Recenteralize();
  bool WriteImgDepthObj(const std::string& filename) const;
  bool WriteImgDepth(const std::string& filename_out);
  bool WriteImgNormal(const std::string& filename_out);
  bool WritePointCloud(const std::string& filename_out);
  // Accessors
  inline const cv::Mat& get_img_depth() const {return img_depth_;}
  inline const Eigen::Vector2i get_offset() const {return offset_;}
  inline void set_img_depth(const cv::Mat& img_depth) {img_depth_ = img_depth;}
  inline void set_offset(const Eigen::Vector2i offset) {offset_ = offset;}

private:
  cv::Mat img_depth_;
  Eigen::Vector2i offset_; // records the change of recenteralize
};

bool ReadH5(const std::string& filename, cv::Mat& img);
bool WriteH5(const std::string& filename, const cv::Mat& img);


} // DPM

#endif  // BASE_LIDAR_H_
