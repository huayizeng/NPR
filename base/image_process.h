#ifndef BASE_IMAGE_PROCESS_H_
#define BASE_IMAGE_PROCESS_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Eigenvalues>

namespace DPM {
void VisualizeDepthImage(const cv::Mat& img_depth, cv::Mat& img_colored);
bool Recenteralize(const float k_margin, cv::Mat& img, Eigen::Vector2i& offset);
bool ComputeNormalImg(const cv::Mat& img_depth, const double kPixelsBetwPts, cv::Mat& img_normal);
bool ComputeGradient(const cv::Mat& img_depth, cv::Mat& img_gradient);
bool SnappingHelper(std::vector<Eigen::Vector2d> &footprint_i_first, std::vector<Eigen::Vector2d> &footprint_i_second);
bool SnappingHelper2(std::vector<Eigen::Vector2d> &footprint_i_first, std::vector<Eigen::Vector2d> &footprint_i_second, std::vector<Eigen::Vector2d> &footprint_i_third);
bool SetMiddleIndexFirst(std::vector<Eigen::Vector2d> &footprint_i_first, std::vector<Eigen::Vector2d> &footprint_i_second, std::vector<Eigen::Vector2d> &footprint_i_third);
bool ComputeXYZN(const cv::Mat& img_depth, const double kPixelsBetwPts, const std::string& path_xyzn);
std::string type2str(int type);
} // DPM

bool ComputeNormalImgF(float *image_depth, double kPixelsBetwPts, int width, int height, unsigned char *image_normal);


#endif  // BASE_IMAGE_PROCESS_H_
