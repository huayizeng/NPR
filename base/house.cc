#include "house.h"

#include <iostream>
#include <boost/filesystem.hpp>
#include <fstream>
#include <limits>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <Eigen/Eigenvalues>

#include "cad.h"
#include "lidar.h"
#include "file_io.h"
#include "image_process.h"
#include "util.h"
#include "mesh.h"

#include "../rapidjson/document.h"
#include "../rapidjson/writer.h"
#include "../rapidjson/stringbuffer.h"
#include "../rapidjson/filereadstream.h"


using Eigen::Vector2d;
using Eigen::Vector3i;
using Eigen::Vector3d;
using std::cerr;
using std::endl;
using std::flush;
using cv::Mat;
using std::ifstream;
using std::istringstream;
using std::numeric_limits;

namespace {

inline double CCW(const Vector2d& p, const Vector2d& q, const Vector2d& s) {
  return ( p[0] * q[1] - p[1] * q[0] + q[0] * s[1] - q[1] * s[0] + s[0] * p[1] - s[1] * p[0]  ) ;
}

inline bool InsideTriangles2d(const Vector2d& v, const Vector2d& v0, const Vector2d& v1, const Vector2d& v2) {
  float a = CCW(v, v0, v1); float b = CCW(v, v1, v2); float c = CCW(v, v2, v0);
  return ( a * b > 0 && b * c > 0 );
}

void FindFiles(const std::string& dir, const std::string& extension, std::vector<std::string>& paths, std::vector<std::string>& filenames) {
  boost::filesystem::path p(dir);
  for (auto i = boost::filesystem::directory_iterator(p); i != boost::filesystem::directory_iterator(); i++) {
    if (boost::filesystem::extension(i->path().filename().string()) == extension) {
      filenames.push_back(i->path().filename().string());
      paths.push_back(i->path().string());
    }
  }
}

void DrawFootprints(const std::vector<Vector2d> footprints, Mat& img_normal) {
  for (unsigned i = 0; i < 4; ++i)
    cv::line(img_normal, cv::Point(floor(footprints[i][0]), floor(footprints[i][1])),
             cv::Point(floor(footprints[(i + 1) % 4][0]), floor(footprints[(i + 1) % 4][1])),
             cv::Scalar( 0, 0, 255 ));
}

void ComputePolygonsMesh(const Mat& img_depth, DPM::Mesh& mesh_polygons) {
  Mat image_depth_cp = img_depth.clone();
  std::vector<Vector3i> faces;
  for (int y = 1; y < image_depth_cp.rows - 1; ++y)
    for (int x = 1; x < image_depth_cp.cols - 1; ++x)
      image_depth_cp.at<float>(y, x) = img_depth.at<float>(y, x) * 0.25 +
                                       img_depth.at<float>(y - 1, x) * 0.125 + img_depth.at<float>(y, x - 1) * 0.125 +
                                       img_depth.at<float>(y + 1, x) * 0.125 + img_depth.at<float>(y, x + 1) * 0.125 +
                                       img_depth.at<float>(y - 1, x - 1) * 0.0625 + img_depth.at<float>(y + 1, x - 1) * 0.0625 +
                                       img_depth.at<float>(y + 1, x + 1) * 0.0625 + img_depth.at<float>(y - 1, x + 1) * 0.0625 ;
  std::vector<Vector3d> vertices_pts;
  for (int y = 0; y < image_depth_cp.rows; ++y)
    for (int x = 0; x < image_depth_cp.cols; ++x)
      vertices_pts.emplace_back(Vector3d(x, y, image_depth_cp.at<float>(y, x)));

  int w = image_depth_cp.cols;
  for (int y = 0; y < image_depth_cp.rows - 1; ++y)
    for (int x = 0; x < image_depth_cp.cols - 1; ++x)
    {
      faces.emplace_back(Vector3i(y * w + x, y * w + x + 1, (y + 1)*w + x));
      faces.emplace_back(Vector3i((y + 1)*w + x + 1, y * w + x + 1, (y + 1)*w + x));
    }
  mesh_polygons.vertices = vertices_pts;
  mesh_polygons.faces = faces;
  return;
}

void ComputePolygonsMeshNoGround(const int kLevel, const Mat& img_depth, DPM::Mesh& mesh_polygons) {
  Mat image_depth_cp = img_depth.clone();
  std::vector<Vector3i> faces;
  for (int y = 1; y < image_depth_cp.rows - 1; ++y)
    for (int x = 1; x < image_depth_cp.cols - 1; ++x)
      image_depth_cp.at<float>(y, x) = img_depth.at<float>(y, x) * 0.25 +
                                       img_depth.at<float>(y - 1, x) * 0.125 + img_depth.at<float>(y, x - 1) * 0.125 +
                                       img_depth.at<float>(y + 1, x) * 0.125 + img_depth.at<float>(y, x + 1) * 0.125 +
                                       img_depth.at<float>(y - 1, x - 1) * 0.0625 + img_depth.at<float>(y + 1, x - 1) * 0.0625 +
                                       img_depth.at<float>(y + 1, x + 1) * 0.0625 + img_depth.at<float>(y - 1, x + 1) * 0.0625 ;
  std::vector<Vector3d> vertices_pts;
  for (int y = 0; y < image_depth_cp.rows; ++y)
    for (int x = 0; x < image_depth_cp.cols; ++x){
      // if (image_depth_cp.at<float>(y, x) < 0.01)
      //   continue;
      vertices_pts.emplace_back(Vector3d(x * kLevel, y * kLevel, image_depth_cp.at<float>(y, x)));
    }

  int w = image_depth_cp.cols;
  for (int y = 0; y < image_depth_cp.rows - 1; ++y)
    for (int x = 0; x < image_depth_cp.cols - 1; ++x)
    {
      if (image_depth_cp.at<float>(y, x) < 0.01 || image_depth_cp.at<float>(y + 1, x) < 0.01 || image_depth_cp.at<float>(y, x + 1) < 0.01)
        continue;
      faces.emplace_back(Vector3i(y * w + x, y * w + x + 1, (y + 1)*w + x));
    }
  for (int y = 0; y < image_depth_cp.rows - 1; ++y)
    for (int x = 0; x < image_depth_cp.cols - 1; ++x)
    {
      if (image_depth_cp.at<float>(y + 1, x) < 0.01 || image_depth_cp.at<float>(y + 1, x + 1) < 0.01 || image_depth_cp.at<float>(y, x + 1) < 0.01)
        continue;
      faces.emplace_back(Vector3i((y + 1)*w + x + 1, y * w + x + 1, (y + 1)*w + x));
    }
      
  mesh_polygons.vertices = vertices_pts;
  mesh_polygons.faces = faces;
  return;
}

} // empty namespace



namespace DPM {

House::House()
  : p_lidar_(std::make_shared<Lidar>()),
    p_cad_(std::make_shared<Cad>()),
    name_("empty")
{}

House::House(const House & house)
  : p_lidar_(std::make_shared<Lidar>(*house.p_lidar_)),
    p_cad_(std::make_shared<Cad>(*house.p_cad_)),
    name_(house.name_),
    feature_dnn_(house.feature_dnn_)
{}

House::~House() = default;
House::House(House&& house)
  : p_lidar_( std::move(house.p_lidar_)),
    p_cad_(std::move(house.p_cad_)),
    name_(std::move(house.name_)),
    feature_dnn_(std::move(house.feature_dnn_))
{}

void House::set_lidar(const Lidar & lidar) {
  std::shared_ptr<Lidar> ptr_temp(std::make_shared<Lidar>(lidar));
  p_lidar_ = std::move(ptr_temp);
}

void House::set_cad(const Cad& cad) {
  std::shared_ptr<Cad> ptr_temp(std::make_shared<Cad>(cad));
  p_cad_ = std::move(ptr_temp);
}

// double House::ComputeDistanceInOne(const Mat& img_gradient) const {
//   std::vector<int> x_request;
//   std::vector<int> y_request;
//   std::vector<double> distance_at_request;
//   Mat img_depth = lidar_.get_img_depth();
//   for (unsigned j = 0; j < img_depth.rows; ++j)
//     for (unsigned i = 0; i < img_depth.cols; ++i) {
//       x_request.push_back(i); y_request.push_back(j);
//     }
//   ComputeDistance(x_request, y_request, distance_at_request);
//   // TODO(Henry): we will clean it but for the moment just leave it this way
//   double sum = 0.0;
//   const double sigma = 45; // best at this moment: 45
//   double sigma2_inv = 1 / (sigma * sigma);
//   for (unsigned i = 0; i < distance_at_request.size(); ++i) {
//     double g = static_cast<double>(img_gradient.at<uchar>(i / img_gradient.rows, i % img_gradient.rows));
//     double temp = distance_at_request[i] * exp(-0.5 * g * g * sigma2_inv);
//     sum += 0.5 * log(temp * temp + 1);
//   }
//   return sum;
// }

bool House::ComputeDistance(const std::vector<int>& x_request, const std::vector<int>& y_request, std::vector<double>& distance_at_request) const {
  if (!p_cad_->is_reconstructed()) {
    cerr << "error: cad not reconstructed" << endl;
    return false;
  }
  Mesh mesh = p_cad_->get_mesh();
  Mat img_surface;
  Mat img_depth = p_lidar_->get_img_depth().clone();
  img_surface.create(img_depth.cols, img_depth.rows, img_depth.type());
  img_surface.setTo(cv::Scalar(0.0));
  std::vector<Vector3d> normals;
  std::vector<Vector3i> faces_no_vertical;

  std::vector<float> depths(img_surface.cols * img_surface.rows, 0.0);

  for (int k = 0; k < mesh.faces.size(); k++) {
    const Vector3d& v0 = mesh.vertices[mesh.faces[k][0]];
    const Vector3d& v1 = mesh.vertices[mesh.faces[k][1]];
    const Vector3d& v2 = mesh.vertices[mesh.faces[k][2]];
    const Vector3d diff1 = v1 - v0;
    const Vector3d diff2 = v2 - v0;
    Vector3d normal = -diff1.cross(diff2);
    normal = normal / normal.norm();
    if (std::abs(normal[2]) < 0.01 )
      continue;
    faces_no_vertical.push_back(mesh.faces[k]);
    normals.push_back(normal);
  }
  // #pragma omp parallel for
  for (int j = 0; j < img_surface.rows; ++j) {
    for (int i = 0; i < img_surface.cols; ++i) {
      float v_z;
      float max_vz = 0.0;
      const Vector2d v(i, j);
      for (int k = 0; k < faces_no_vertical.size(); k++) {
        const Vector3d& v0 = mesh.vertices[faces_no_vertical[k][0]];
        const Vector3d& v1 = mesh.vertices[faces_no_vertical[k][1]];
        const Vector3d& v2 = mesh.vertices[faces_no_vertical[k][2]];
        const Vector2d v00(v0[0], v0[1]);
        const Vector2d v10(v1[0], v1[1]);
        const Vector2d v20(v2[0], v2[1]);
        // #pragma omp critical
        if (InsideTriangles2d(v, v00, v10, v20)) {
          if (normals[k][2] == 0)
            v_z = v0[2];
          else
            v_z = -((v[1] - v0[1]) * normals[k][1] + (v[0] - v0[0]) * normals[k][0]) / normals[k][2] + v0[2];
          max_vz = std::max(v_z, max_vz);
        }
      }
      img_surface.at<float>(j, i) = max_vz;
    }
  }

  // Mat img_colored1, img_colored2;
  // VisualizeDepthImage(img_depth, img_colored1);
  // VisualizeDepthImage(img_surface, img_colored2);
  // cv::imshow("1", img_colored1);
  // cv::waitKey();
  // cv::imshow("2", img_colored2);
  // cv::imwrite("img_surface.png", img_colored2);
  Mat diff;
  cv::absdiff(img_depth, img_surface, diff);

  diff *= 10;
  for (unsigned i = 0; i < x_request.size(); ++i)
    distance_at_request.push_back(static_cast<double>(diff.at<float>(y_request[i], x_request[i])));
  // Mat img_normal_1;
  // Mat img_normal_2;
  // ComputeNormalImg(img_surface, 0.4, img_normal_1);
  // cv::imshow(",", img_normal_1);
  // cv::waitKey();
  // ComputeNormalImg(img_depth, 0.4, img_normal_2);
  // cv::imshow(",", img_normal_2);
  // cv::waitKey();
  return true;
}
  bool House::ComputeImgSurface(Mat &img_surface) const {
    if (!p_cad_->is_reconstructed()) {
      cerr << "error: cad not reconstructed" << endl;
      return false;
    }
    Mesh mesh = p_cad_->get_mesh();
    Mat img_depth = p_lidar_->get_img_depth().clone();
    img_surface.create(img_depth.cols, img_depth.rows, img_depth.type());
    img_surface.setTo(cv::Scalar(0.0));
    std::vector<Vector3d> normals;
    std::vector<Vector3i> faces_no_vertical;

    std::vector<float> depths(img_surface.cols * img_surface.rows, 0.0);

    for (int k = 0; k < mesh.faces.size(); k++) {
      const Vector3d& v0 = mesh.vertices[mesh.faces[k][0]];
      const Vector3d& v1 = mesh.vertices[mesh.faces[k][1]];
      const Vector3d& v2 = mesh.vertices[mesh.faces[k][2]];
      const Vector3d diff1 = v1 - v0;
      const Vector3d diff2 = v2 - v0;
      Vector3d normal = -diff1.cross(diff2);
      normal = normal / normal.norm();
      if (std::abs(normal[2]) < 0.01 )
	continue;
      faces_no_vertical.push_back(mesh.faces[k]);
      normals.push_back(normal);
    }
    // #pragma omp parallel for
    for (int j = 0; j < img_surface.rows; ++j) {
      for (int i = 0; i < img_surface.cols; ++i) {
	float v_z;
	float max_vz = 0.0;
	const Vector2d v(i, j);
	for (int k = 0; k < faces_no_vertical.size(); k++) {
	  const Vector3d& v0 = mesh.vertices[faces_no_vertical[k][0]];
	  const Vector3d& v1 = mesh.vertices[faces_no_vertical[k][1]];
	  const Vector3d& v2 = mesh.vertices[faces_no_vertical[k][2]];
	  const Vector2d v00(v0[0], v0[1]);
	  const Vector2d v10(v1[0], v1[1]);
	  const Vector2d v20(v2[0], v2[1]);
	  // #pragma omp critical
	  if (InsideTriangles2d(v, v00, v10, v20)) {
	    if (normals[k][2] == 0)
	      v_z = v0[2];
	    else
	      v_z = -((v[1] - v0[1]) * normals[k][1] + (v[0] - v0[0]) * normals[k][0]) / normals[k][2] + v0[2];
	    max_vz = std::max(v_z, max_vz);
	  }
	}
	img_surface.at<float>(j, i) = max_vz;
      }
    }
    return true;
  }
  static bool overlap(const Eigen::Vector3i &tri1, const Eigen::Vector3i &tri2) {
    int count = 0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
	if (tri1[i] == tri2[j])
	  count++;
    return count == 2 ? true : false;
  }
  static bool check_same_plane(const Eigen::Vector3i &face1,
			       const Eigen::Vector3i &face2,
			       const std::vector<Eigen::Vector3d> vertices) {
    if (overlap(face1, face2)) {
      Eigen::Vector3d normal1 = (vertices[face1[1]] - vertices[face1[0]]).cross(vertices[face1[2]] - vertices[face1[0]]);
      Eigen::Vector3d normal2 = (vertices[face2[1]] - vertices[face2[0]]).cross(vertices[face2[2]] - vertices[face2[0]]);
      if (std::abs(normal1.dot(normal2) / (normal1.norm() * normal2.norm())) > 0.99) {
	return true;
      } else {
	return false;
      }
    } else {
      return false;
    }
  }
  bool House::generate_face_to_same_plane() {
    const cv::Mat &depth = p_lidar_->get_img_depth();
    Eigen::Vector2i img_size = Eigen::Vector2i(depth.rows, depth.cols);
    const Mesh &mesh = get_cad().get_mesh();
    if (img_size[0] == 0 && img_size[1] == 0) {
      //uninitialized
      return false;
    }

    same_plane_map.clear();
    for (unsigned char i = 0; i <= mesh.faces.size(); i++) {
      same_plane_map[i] = i;
    }
    for (unsigned char i = 0; i < mesh.faces.size(); i++) {
      for (unsigned char j = 0; j < mesh.faces.size(); j++) {
	if (check_same_plane(mesh.faces[i], mesh.faces[j], mesh.vertices)) {
	  //                    std::cout << "same plane: "<< (int)(j+1) << " is: " << (int)(i+1) << endl;
	  same_plane_map[j + 1] = same_plane_map[i + 1];
	  //                    std::cout << "same plane map: "<< (int)(j+1) << " is: " << (int)facemap[i+1] << endl;
	}
      }
    }
    return true;
  }
  bool House::generate_owner_and_surface_of_face_helper() {
    const cv::Mat &depth = p_lidar_->get_img_depth();
    Eigen::Vector2i img_size = Eigen::Vector2i(depth.rows, depth.cols);
    if (img_size[0] == 0 && img_size[1] == 0) {
      //uninitialized
      return false;
    }
    // float kStepSize = 0.75;
    const Mesh &mesh = get_cad().get_mesh();
    img_surface.create(img_size[1], img_size[0], CV_32F);
    img_surface.setTo(cv::Scalar(0.0));
    face_owner.create(img_size[1], img_size[0], CV_8UC1);
    face_owner.setTo(cv::Scalar(0));
    std::vector<Eigen::Vector3d> normals;
    std::vector<Eigen::Vector3i> faces_no_vertical;
    std::vector<int> original_face_indices;

    for (int k = 0; k < mesh.faces.size(); k++) {
      const Eigen::Vector3d& v0 = mesh.vertices[mesh.faces[k][0]];
      const Eigen::Vector3d& v1 = mesh.vertices[mesh.faces[k][1]];
      const Eigen::Vector3d& v2 = mesh.vertices[mesh.faces[k][2]];
      const Eigen::Vector3d diff1 = v1 - v0;
      const Eigen::Vector3d diff2 = v2 - v0;
      Eigen::Vector3d normal = -diff1.cross(diff2);
      normal = normal / normal.norm();
      if (std::abs(normal[2]) < 0.01 )
	continue;
      faces_no_vertical.push_back(mesh.faces[k]);
      original_face_indices.push_back(k);
      normals.push_back(normal);
    }
    // #pragma omp parallel for
    for (int j = 0; j < img_surface.rows; ++j) {
      for (int i = 0; i < img_surface.cols; ++i) {
	float v_z;
	float max_vz = 0.0;
	int face_owner_idx = 0;
	const Eigen::Vector2d v(i, j);
	for (int k = 0; k < faces_no_vertical.size(); k++) {
	  const Eigen::Vector3d& v0 = mesh.vertices[faces_no_vertical[k][0]];
	  const Eigen::Vector3d& v1 = mesh.vertices[faces_no_vertical[k][1]];
	  const Eigen::Vector3d& v2 = mesh.vertices[faces_no_vertical[k][2]];
	  const Eigen::Vector2d v00(v0[0], v0[1]);
	  const Eigen::Vector2d v10(v1[0], v1[1]);
	  const Eigen::Vector2d v20(v2[0], v2[1]);
	  // #pragma omp critical
	  if (InsideTriangles2d(v, v00, v10, v20)) {
	    if (normals[k][2] == 0)
	      v_z = v0[2];
	    else
	      v_z = -((v[1] - v0[1]) * normals[k][1] + (v[0] - v0[0]) * normals[k][0]) / normals[k][2] + v0[2];
	    if(v_z > max_vz){
	      face_owner_idx = original_face_indices[k]+1;
	    }
	    max_vz = std::max(v_z, max_vz);
	  }
	}
	img_surface.at<float>(j, i) = max_vz;
	face_owner.at<unsigned char>(j, i) = face_owner_idx;
      }
    }
    return true;
  }
bool Preprocess(FileIO & file_io, HouseGroup & housegroup) {

  // /* Read DSM and DTM */
  // std::vector<std::string> paths_dsm;
  // std::vector<std::string> filenames_dsm;
  // FindFiles(file_io.GetDSMRoot(), ".hdf5", paths_dsm, filenames_dsm);
  // std::vector<std::string> paths_dtm;
  // std::vector<std::string> filenames_dtm;
  // FindFiles(file_io.GetDTMRoot(), ".hdf5", paths_dtm, filenames_dtm);
  // assert(paths_dsm.size() == paths_dtm.size());
  // int kLowestHeight = 2.0;
  // for (unsigned i_lidar = 0; i_lidar < paths_dtm.size(); ++i_lidar) {
  //   // if (
  //   //     filenames_dtm[i_lidar] != "se1010_0_12_19_mat.hdf5" 
  //   // )
  //   //   continue;
  //   if (filenames_dtm[i_lidar] != filenames_dsm[i_lidar]) {
  //     cerr << "name of dsm differs from dtm" << endl;
  //     return false;
  //   }
  //   Lidar lidar;
  //   lidar.ReadImgDepth(paths_dsm[i_lidar], paths_dtm[i_lidar]);
  //   lidar.RemoveLowerPoints(kLowestHeight);
  //   if (!lidar.KeepCenterConnectedComponent()) {
  //     cerr << "error when running KeepCenterConnectedComponent" << endl;
  //     return false;
  //   }
  //   if (!lidar.Recenteralize()) {
  //     cerr << "error when running Recenteralize" << endl;
  //     return false;
  //   }
  //   House house;
  //   house.set_lidar(lidar);
  //   house.set_name(filenames_dtm[i_lidar]);
  //   housegroup.houses.push_back(house);
  //   // Mat img_colored;
  //   // lidar.GetColoredDepthImage(img_colored);
  //   // cv::imshow(",", img_colored);
  //   // cv::waitKey();
  // }
  // return true;

  // /* Read diff */
  std::vector<std::string> paths_diff;
  std::vector<std::string> filenames_diff;
  FindFiles(file_io.GetDiffRoot(), ".hdf5", paths_diff, filenames_diff);
  // bool flag = false;
  for (unsigned i_lidar = 0; i_lidar < paths_diff.size(); ++i_lidar) {
    // if(i_lidar <= 10) {
    //   continue;
    // }
    // if(i_lidar >= 20) {
    //   break;
    // }
    // if (filenames_diff[i_lidar] != "sz2595_17_5_19_mat.hdf5")
    //   continue;
    // if (filenames_diff[i_lidar] == "800691.hdf5")
    //   flag = true;
    // if(flag == false) {
    //   continue;
    // }
    Lidar lidar;
    lidar.ReadImgDepth(paths_diff[i_lidar]);
    House house;
    house.set_lidar(lidar);
    house.set_name(filenames_diff[i_lidar]);
    std::string dir_dnn_features = file_io.GetDNNFeatures();

    auto name_wo_hdf5 = house.get_name();
    name_wo_hdf5.erase(name_wo_hdf5.end()-5, name_wo_hdf5.end());
    auto path_dnn_feature = dir_dnn_features + "/" + name_wo_hdf5 + ".pb";
    // std::fstream input(path_dnn_feature, std::ios::in | std::ios::binary);
    // if (!input) {
    //   cerr << "preprocess: no pb file: " << path_dnn_feature << endl;
    //   continue;
    // }
    // input.close();
    housegroup.houses.push_back(house);
  }
  cerr << "finish reading diff files" << endl;
  cerr << "finish Preprocess" << endl;
  return true;
}

bool GetDNNPrediction(FileIO& file_io, HouseGroup& housegroup) {
  std::string dir_dnn_features = file_io.GetDNNFeatures();
  for (auto && house : housegroup.houses) {
    auto name_wo_hdf5 = house.get_name();
    name_wo_hdf5.erase(name_wo_hdf5.end()-5, name_wo_hdf5.end());
    auto path_dnn_feature = dir_dnn_features + "/" + name_wo_hdf5 + ".pb";
    FeatureDNNProto feature_dnn;
    std::fstream input(path_dnn_feature, std::ios::in | std::ios::binary);
    if (!input) {
      cerr << path_dnn_feature << ": File not found." << endl;
    } else if (!feature_dnn.ParseFromIstream(&input)) {
      cerr << path_dnn_feature << ": Failed to parse the file." << endl;
      return -1;
    }    
    house.set_feature_dnn(feature_dnn);
    cerr <<"#dormers: " << house.get_feature_dnn().dormer_x_size() << endl;
    cerr <<"#chimneys: " << house.get_feature_dnn().chimney_x_size() << endl;
  }
}

void WriteLidarImg(const std::string& dir_depth_img, const std::string& dir_normal_img, HouseGroup & housegroup) {
  cerr << "Start WriteLidarImg... " << endl;
  for (auto && house : housegroup.houses) {
    Lidar lidar = house.get_lidar();
    lidar.WriteImgDepth(dir_depth_img + "/" + house.get_name() + ".h5");
    lidar.WriteImgNormal(dir_normal_img + "/" + house.get_name() + ".png");
  }
}

void WriteXYZN(const std::string& dir_xyzn, const HouseGroup & housegroup) {
  for (auto && house : housegroup.houses) {
    Lidar lidar = house.get_lidar();
    ComputeXYZN(lidar.get_img_depth(), 1.0, dir_xyzn + "/" + house.get_name() + ".xyzn");
  }
}

void WritePointCloud(FileIO & file_io, HouseGroup & housegroup) {
  std::string dir_point_cloud = file_io.GetPointCloudDir();
  for (auto && house : housegroup.houses) {
    Lidar lidar = house.get_lidar();
    lidar.WritePointCloud(dir_point_cloud + "/" + house.get_name() + ".obj");
  }
}

// TODO(Henry): this function should be re-written as recognition and compute paras from root to leaf
bool ComputeCad(FileIO & file_io, HouseGroup & housegroup) {
  cerr << "num of houses before recognition and reconstruction: " << housegroup.houses.size() << endl;
  int n_lost_recognition = 0;
  HouseGroup housegroup_new;
  for (auto && house : housegroup.houses) {
    // OptRecog opt_recog(house.get_lidar().get_img_depth(), house.get_name(), file_io);
    // cerr << "process house: " << house.get_name() << endl;
    OptRecog opt_recog(house.get_lidar().get_img_depth(), house.get_name(), file_io, house.get_feature_dnn());
    Cad cad = house.get_cad();
    if (!cad.Recognition(opt_recog)){
      // GOT_HERE; cerr << "not recognizable" << endl;
      n_lost_recognition ++;
      continue;
    }
    bool LandU = false;
    auto house_type = opt_recog.feature_dnn.housetype();

    // TODO(We ran i shape because we want to find some complicated examples)
    // if(house_type == 3) 
    //   continue;

    if(house_type == 5 || house_type == 6)
      LandU = true;
    cad.ReconstuctAll(LandU);
    house.set_cad(cad);
    // cerr << "compute cad: dormer size now" << house.get_feature_dnn().dormer_x_size() << endl;

    housegroup_new.houses.push_back(house);
    // WARINING: delete it

    // Mat img_normal;
    // DPM::ComputeNormalImg(house.get_lidar().get_img_depth(), 0.4, img_normal);
    // DrawFootprints(dynamic_cast<TwoIModule*>(cad.p_module_root_.get())->footprint_i_first_, img_normal);
    // DrawFootprints(dynamic_cast<TwoIModule*>(cad.p_module_root_.get())->footprint_i_second_, img_normal);
    // ModuleType module_recognized = cad.get_module_type();
    // cerr << static_cast<std::underlying_type<ModuleType>::type>(module_recognized) << endl;
  }
  housegroup = housegroup_new;
  cerr << "house lost during reconstruction: " << n_lost_recognition << endl;
  cerr << "num of houses AFTER recognition and reconstruction: " << housegroup.houses.size() << endl;

  return true;
}

void WritePolygonModel(const std::string& dir_polygon_model, HouseGroup & housegroup) {
  for (auto && house : housegroup.houses) {
    cerr << "WritePolygonModel, house name: " << house.get_name() << endl;
    Lidar lidar = house.get_lidar();
    lidar.WriteImgDepthObj(dir_polygon_model + "/" + house.get_name() + ".obj");
  }
}

void WriteCad(const std::string& dir_cad, HouseGroup & housegroup) {
  for (auto && house : housegroup.houses) {
    if (!house.get_cad().is_reconstructed())
      continue;
    // cerr << "WriteCad: " << dir_cad + "/" + house.get_name() + ".obj" << endl;
    Cad cad = house.get_cad();
    cad.WriteMesh(dir_cad + "/" + house.get_name() + ".obj");
    // cad.WriteMesh("/Users/huayizeng/Desktop/temp/1s.obj");
  }
}

void WriteStates(const std::string& dir_states, HouseGroup & housegroup) {
  for (auto && house : housegroup.houses){
    // cerr << dir_states + "/" + house.get_name() + ".json" << endl;
    house.write_state(dir_states + "/" + house.get_name() + ".json");
  }
}

bool ReadOffset(const std::string& path_offset, int & coord_tl_x, int & coord_tl_y) {
    coord_tl_x = 0; coord_tl_y = 0;
    ifstream ifstr;
    ifstr.open(path_offset.c_str());
    if (!ifstr.is_open()) {
      cerr << "Cannot open a file: " << path_offset << endl;
      return false;
    }
    std::string line;
    float temp = 0;
    getline(ifstr, line); istringstream isstr(line); isstr >> temp; coord_tl_x += int(temp); 
    getline(ifstr, line); istringstream isstr2(line); isstr2 >> temp; coord_tl_y += int(temp);
    getline(ifstr, line); istringstream isstr3(line); isstr3 >> temp; coord_tl_x += int(temp);
    getline(ifstr, line); istringstream isstr4(line); isstr4 >> temp; coord_tl_y += int(temp);
    return true;
}

void WriteLatLng(HouseGroup & housegroup, FileIO & file_io) {
  auto dir_offset = file_io.GetOffsetDir();
  MeshGroup mesh_group;
  cerr << "start WriteLatLng" << endl;
  for (unsigned ii = 0; ii < housegroup.houses.size(); ++ii) {    
    /* Read from already-optimized, this could be seperated into one single function */
    // cerr << ii << endl;
    // // if (filenames_opt_cad[ii] != "se1010_0_9_21_mat.hdf5.obj" || filenames_opt_cad[ii] != "se1010_16_2_32_mat.hdf5.obj") continue;
    // auto paths_opt_cad = file_io.GetCadOptimizedDir() + "/" + housegroup.houses[ii].get_name() + ".obj";
    // cerr << paths_opt_cad << endl;
    // Mesh mesh;
    // mesh.Read(paths_opt_cad);
    // Mat img_depth;
    // std::string filename_pts(file_io.GetDepthImgDir() + "/" + split_string(filenames_opt_cad[ii], ".obj")[0] + ".h5");
    // std::string filename_pts("fd");
    // ReadH5(filename_pts, img_depth);
    House house = housegroup.houses[ii];
    Lidar lidar = house.get_lidar();
    Mesh mesh = house.get_cad().get_mesh();
    
    int coord_tl_x = 0; int coord_tl_y = 0;
    auto path_offset = dir_offset + "/" + split_string(housegroup.houses[ii].get_name(), ".hdf5")[0] + ".txt";
    ReadOffset(path_offset, coord_tl_x, coord_tl_y);
    // coord_tl_x += lidar.get_offset()[0]; coord_tl_y += lidar.get_offset()[1];
    Mesh mesh_offset;
    for (auto && v : mesh.vertices)
      mesh_offset.vertices.push_back(Vector3d(v[0] + coord_tl_x , (v[1] + coord_tl_y ) , v[2]));
    mesh_offset.faces = mesh.faces;
    mesh_group.meshes.push_back(mesh_offset);

  }
  mesh_group.Write("ourhouses_se3500.obj");
  // mesh_group.WriteData("ourhouses_wo_opt_region.txt");
  return;
}

void House::write_state(const std::string& path) {
  auto footprints = p_cad_->get_footprints();
  auto dormers = p_cad_->get_dormers();
  auto chimneys = p_cad_->get_chimneys();
  auto angle  = p_cad_->get_azimuth();
  rapidjson::Document doc;
  doc.SetObject();
  rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();
  

  doc.AddMember("angle", angle, allocator);
   
  rapidjson::Value footprint_x_obj(rapidjson::kArrayType);
  rapidjson::Value footprint_y_obj(rapidjson::kArrayType);
  for(auto&& footprint : footprints) {
    for(auto&& v : footprint) {
      footprint_x_obj.PushBack(v[0], allocator); 
      footprint_y_obj.PushBack(v[1], allocator);
    }
  }
  doc.AddMember("footprint_x", footprint_x_obj, allocator);
  doc.AddMember("footprint_y", footprint_y_obj, allocator);

  rapidjson::Value dormers_x(rapidjson::kArrayType);
  rapidjson::Value dormers_y(rapidjson::kArrayType);
  for(auto&& coor : dormers) {
    dormers_x.PushBack(coor[0], allocator);
    dormers_y.PushBack(coor[1], allocator);
  }
  doc.AddMember("dormers_x", dormers_x, allocator);
  doc.AddMember("dormers_y", dormers_y, allocator);

  rapidjson::Value chimneys_x(rapidjson::kArrayType);
  rapidjson::Value chimneys_y(rapidjson::kArrayType);
  for(auto&& coor : chimneys) {
    chimneys_x.PushBack(coor[0], allocator);
    chimneys_y.PushBack(coor[1], allocator);
  }
  doc.AddMember("chimneys_x", chimneys_x, allocator);
  doc.AddMember("chimneys_y", chimneys_y, allocator);

  // Stringify the DOM
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);
  std::cout << buffer.GetString() << std::endl;

  FILE* file = fopen(path.c_str(), "w"); 
  if (file)  
  {  
    fputs(buffer.GetString(),file);  
    fclose(file);  
  }
}

#ifdef GPU_FOUND

bool House::ComputeDistanceGPU(const std::vector<int>& x_request, const std::vector<int>& y_request,
                               std::vector<double>& distance_at_request, Corender& mycorender) const {
  // TODO(Henry): check if mycorender has successfully been initialized
  if (!p_cad_->is_reconstructed()) {
    cerr << "error: cad not reconstructed" << endl;
    return false;
  }
  Mesh mesh = p_cad_->get_mesh();
  Mat img_surface;
  Mat img_depth = p_lidar_->get_img_depth().clone();
  img_surface.create(img_depth.cols, img_depth.rows, img_depth.type());
  img_surface.setTo(cv::Scalar(0.0));
  std::vector<Vector3d> normals;
  std::vector<Vector3i> faces_no_vertical;

  std::vector<float> depths(img_surface.cols * img_surface.rows, 0.0);

  const float near = 0.0f;
  const float far = 100.0f;
  if (!mycorender.renderDepth(mesh, near, far, img_surface.cols, img_surface.rows, depths))
    return false;
  for (unsigned j = 0; j < img_surface.rows; ++j)
    for (unsigned i = 0; i < img_surface.cols; ++i)
      img_surface.at<float>(img_surface.rows - j - 1, i) = depths[j * img_surface.cols + i];
  Mat diff;
  cv::absdiff(img_depth, img_surface, diff);

  diff *= 10;
  for (unsigned i = 0; i < x_request.size(); ++i)
    distance_at_request.push_back(static_cast<double>(diff.at<float>(y_request[i], x_request[i])));
  return true;
}

void RenderForSketchUp(FileIO & file_io) {
  std::vector<std::string> paths_opt_cad;
  std::vector<std::string> filenames_opt_cad;
  FindFiles(file_io.GetCadOptimizedDir(), ".obj", paths_opt_cad, filenames_opt_cad);
  int nrows_houses = 31;
  int ncols_houses = 26;
  Mesh supermesh;
  Mesh supermesh_polygons;
  int size_grid = 200;
  int kLevel = 2;
  for (unsigned ii = 0; ii < paths_opt_cad.size(); ++ii) {
    // if (ii > 100) break;
    // if (filenames_opt_cad[ii] != "se1010_0_9_21_mat.hdf5.obj" || filenames_opt_cad[ii] != "se1010_16_2_32_mat.hdf5.obj") continue;
    Mesh mesh;
    mesh.Read(paths_opt_cad[ii]);
    Corender mycorender;
    Mat img_depth;
    std::string filename_pts(file_io.GetDepthImgDir() + "/" + split_string(filenames_opt_cad[ii], ".obj")[0] + ".h5");
    // std::string filename_pts("fd");
    ReadH5(filename_pts, img_depth);
    if (ii >= nrows_houses * ncols_houses)
      break;
    float kScale = static_cast<float>(size_grid) / img_depth.rows;
    for(auto&& ele : mesh.vertices) {
      ele = ele * kScale + Vector3d((ii % ncols_houses) * size_grid * 2, (ii / ncols_houses) * size_grid, 0);
    }
    
    cv::resize(img_depth, img_depth, cv::Size(), 1.0 / static_cast<double>(kLevel), 1.0 / static_cast<double>(kLevel));
    Mesh mesh_polygons;
    // if ((ii % ncols_houses) < 15 && (ii % ncols_houses) > 4 && (ii / ncols_houses) < 15 && (ii / ncols_houses) > 4)
    if ((ii % ncols_houses) < 21 && (ii % ncols_houses) > 0 && (ii / ncols_houses) < 21 && (ii / ncols_houses) > 0)
    // if (true)
    {
      ComputePolygonsMeshNoGround(kLevel, img_depth, mesh_polygons);
      for(auto&& ele : mesh_polygons.vertices) {
        ele = ele * kScale + Vector3d((ii % ncols_houses) * size_grid * 2, (ii / ncols_houses) * size_grid, 0);
      }
      supermesh_polygons.Merge(mesh_polygons);
    }

    supermesh.Merge(mesh);
  }
  supermesh.Write("supermesh.obj");
  supermesh_polygons.Write("supermesh_polygons.obj");
}
// void RenderForMapMode() {
//   MeshGroup mesh_group;
//   mesh_group.ReadData("ourhouses_wo_opt.txt");
//   MeshGroup mesh_group_2i;
//   mesh_group_2i.ReadData("ourhouses_wo_opt_2i.txt");
//   mesh_group.meshes.insert(mesh_group.meshes.end(), mesh_group_2i.meshes.begin(), mesh_group_2i.meshes.end());
//   MeshGroup mesh_group_new;
//   const int x_start = 500; const int x_range = 800; const int y_start = 1000; const int y_range = 1000; 
//   for (auto && mesh : mesh_group.meshes) {
//     for(auto&& v : mesh.vertices){
//       v[0] -= x_start;
//       v[1] = 20000 - v[1];
//       v[1] -= y_start;
//     }
//     Vector2d ranges[3];
//     for (int a = 0; a < 3; ++a) {
//       ranges[a][0] = numeric_limits<double>::max(); ranges[a][1] = - numeric_limits<double>::max();
//     }
//     for (const auto& face : mesh.faces)
//       for (int i = 0; i < 3; ++i)
//         for (int a = 0; a < 3; ++a) {
//           ranges[a][0] = std::min(ranges[a][0], mesh.vertices[face[i]][a]); ranges[a][1] = std::max(ranges[a][1], mesh.vertices[face[i]][a]);
//         }
//     if (ranges[0][0] > 0 && ranges[0][1] < x_range && ranges[1][0] > 0 && ranges[1][1] < y_range)
//       mesh_group_new.meshes.push_back(mesh);
//   }
//   Corender mycorender;
//   const float near = 0.0f;; const float far = 100.0f; const int width = x_range; const int height = y_range;
//   // mycorender.initGL(near, far, width, height);
//   mycorender.initGLMapmMode(width, height, near, far);
//   std::vector<int> fids;
//   mycorender.MapModeVisulization(mesh_group, near, far - 1, width, height, fids);
//   Mat img_rendered;
//   img_rendered.create(y_range, x_range, CV_8UC3);
//   img_rendered.setTo(cv::Vec3b(0, 0, 0));
//   int ind = 0;
//   for (unsigned j = 0; j < img_rendered.rows; ++j) {
//     for (unsigned i = 0; i < img_rendered.cols; ++i) {
//       int f = fids[ind];
//       // unsigned char a = (unsigned char)(f % 256);
//       unsigned char b = (unsigned char)((f >> 8) % 256);
//       unsigned char g = (unsigned char)((f >> 16) % 256);
//       unsigned char r = (unsigned char)((f >> 24) % 256);
//       img_rendered.at<cv::Vec3b>(img_rendered.rows - j - 1, i) = cv::Vec3b(b, g, r);
//       ind++;
//     }
//   }
//   mycorender.destroy();

//   cv::imwrite("mesa_map_mode.png", img_rendered);
//   cerr << "write mesa_map_mode.png" << endl;
// }

void RenderOneMeshPolygon(const Mesh& mesh, const Mesh& mesh_polygons, const Mat& img_depth, const float kTimes, const int mode, Mat& img_rendered) {
    const float near = 0.0f; const float far = 100.0f;
    std::vector<float> depths; std::vector<int> fids;
    Corender mycorender;
    mycorender.initGLHighResolution(mesh, near, far, img_depth.cols * kTimes, img_depth.rows * kTimes,
                                    img_depth.cols, img_depth.rows,
                                    img_depth.cols / 2, img_depth.rows / 2);
    mycorender.renderFidDepthOverlap(mesh, mesh_polygons, near, far, img_depth.cols * kTimes, 
                                    img_depth.rows * kTimes, img_depth.cols / 2, 
                                    img_depth.rows / 2, mode, depths, fids );

    cerr << "depths.size: " << depths.size() << endl;
    int ind = 0;
    for (unsigned j = 0; j < img_rendered.rows; ++j) {
      for (unsigned i = 0; i < img_rendered.cols; ++i) {
        int f = fids[ind];
        unsigned char b = (unsigned char)((f >> 8) % 256);
        unsigned char g = (unsigned char)((f >> 16) % 256);
        unsigned char r = (unsigned char)((f >> 24) % 256);
        img_rendered.at<cv::Vec3b>(img_rendered.rows - j - 1, i) = cv::Vec3b(b, g, r);
        ind++;
      }
    }
    mycorender.destroy();  
}

void RenderOneMesh(const Mesh& mesh, const Mesh& mesh_polygons, const Mat& img_depth, 
                  Mat& img_rendered_polygon, Mat& img_rendered_cad, Mat& img_rendered_overlap) {
    const float kTimes = 8;
    img_rendered_polygon.create(img_depth.rows * kTimes, img_depth.cols * kTimes, CV_8UC3);
    img_rendered_polygon.setTo(cv::Vec3b(0, 0, 0));
    img_rendered_cad = img_rendered_polygon.clone();
    img_rendered_overlap = img_rendered_polygon.clone();
    cerr << "img_depth.cols * kTimes: " << img_depth.cols * kTimes << endl;
    RenderOneMeshPolygon(mesh, mesh_polygons, img_depth, kTimes, 2, img_rendered_polygon);
    RenderOneMeshPolygon(mesh, mesh_polygons, img_depth, kTimes, 1, img_rendered_cad);
    RenderOneMeshPolygon(mesh, mesh_polygons, img_depth, kTimes, 0, img_rendered_overlap);
}

void RenderForDebug(FileIO & file_io, HouseGroup & housegroup) {
  cerr << "start RenderForDebug" << endl;
  for (auto && house : housegroup.houses) {
    auto mesh = house.get_cad().get_mesh();
    Mat img_depth = house.get_lidar().get_img_depth();
    Mesh mesh_polygons; ComputePolygonsMesh(img_depth, mesh_polygons);
    Mat img_rendered_polygon, img_rendered_cad, img_rendered_overlap;
    RenderOneMesh(mesh, mesh_polygons, img_depth, img_rendered_polygon, img_rendered_cad, img_rendered_overlap);
    cv::imwrite("/home/huayizeng/img_rendered/" + house.get_name() + "_img_rendered_polygon.png", img_rendered_polygon);
    cv::imwrite("/home/huayizeng/img_rendered/" + house.get_name() + "_img_rendered_cad.png", img_rendered_cad);
    cv::imwrite("/home/huayizeng/img_rendered/" + house.get_name() + "_img_rendered_overlap.png", img_rendered_overlap);
  }
}

void RenderForDebugFromOptCadDir(FileIO & file_io) {
  // Read the houses first
  // WARNING: Not getting image from housegroup, but instead from opt_cad
  cerr << "start RenderForDebugFromOptCadDir" << endl;
  std::vector<std::string> paths_opt_cad;
  std::vector<std::string> filenames_opt_cad;
  FindFiles(file_io.GetCadOptimizedDir(), ".obj", paths_opt_cad, filenames_opt_cad);
  for (unsigned ii = 0; ii < paths_opt_cad.size(); ++ii) {
    // cerr << ii << endl;
    // if (ii > 100) break;
    // if (filenames_opt_cad[ii] != "se1010_0_9_21_mat.hdf5.obj" || filenames_opt_cad[ii] != "se1010_16_2_32_mat.hdf5.obj") continue;
    Mesh mesh; mesh.Read(paths_opt_cad[ii]);
    Mat img_depth;
    std::string filename_pts(file_io.GetDepthImgDir() + "/" + split_string(filenames_opt_cad[ii], ".obj")[0] + ".h5");
    // std::string filename_pts("fd");
    ReadH5(filename_pts, img_depth);
    Mesh mesh_polygons;
    ComputePolygonsMesh(img_depth, mesh_polygons);
    Mat img_rendered_polygon, img_rendered_cad, img_rendered_overlap;
    RenderOneMesh(mesh, mesh_polygons, img_depth, img_rendered_polygon, img_rendered_cad, img_rendered_overlap);
    cv::imwrite("/home/huayizeng/img_rendered/" + filenames_opt_cad[ii] + "_img_rendered_polygon.png", img_rendered_polygon);
    cv::imwrite("/home/huayizeng/img_rendered/" + filenames_opt_cad[ii] + "_img_rendered_cad.png", img_rendered_cad);
    cv::imwrite("/home/huayizeng/img_rendered/" + filenames_opt_cad[ii] + "_img_rendered_overlap.png", img_rendered_overlap);
  }
}
#endif

void ReadHouseFromSavedDir(FileIO & file_io, HouseGroup & housegroup) {
  std::vector<std::string> paths_opt_cad;
  std::vector<std::string> filenames_opt_cad;
  housegroup.houses.clear();
  FindFiles(file_io.GetCadSavedDir(), ".obj", paths_opt_cad, filenames_opt_cad);
  for (unsigned ii = 0; ii < paths_opt_cad.size(); ++ii) {
    Mesh mesh;
    mesh.Read(paths_opt_cad[ii]);
    House house;
    Cad cad = house.get_cad();
    cad.set_mesh(mesh);
    cad.set_is_reconstructed(true);
    house.set_name(std::to_string(ii));
    house.set_cad(cad);
    housegroup.houses.push_back(house);
  }
}

void AddOverhangs(HouseGroup & housegroup) {
  HouseGroup housegroup_new;
  for (auto && house : housegroup.houses) {

    TwoIModule twoiModule;
    // PiModule threeiModule;
    // LModule lModule;
    
    Cad cad = house.get_cad();
    twoiModule.imodule_first_.rooftype_ = RoofType::kGable; twoiModule.imodule_second_.rooftype_ = RoofType::kGable; 
    // threeiModule.imodule_first_.rooftype_ = RoofType::kHip; threeiModule.imodule_second_.rooftype_ = RoofType::kHip; threeiModule.imodule_third_.rooftype_ = RoofType::kHip; 
    // lModule.imodule_first_.rooftype_ = RoofType::kGable; lModule.imodule_second_.rooftype_ = RoofType::kGable;

    cad.p_module_root_ = std::make_shared<TwoIModule>(twoiModule);
    // cad.p_module_root_ = std::make_shared<PiModule>(threeiModule);
    // cad.p_module_root_ = std::make_shared<LModule>(lModule);

    cerr << static_cast<std::underlying_type<ModuleType>::type>(cad.get_module_type()) << endl; 
    // if (cad.get_module_type() != DPM::ModuleType::kTwoIModule)
    //   continue;
    cerr << "start adding overhang" << endl;
    cad.AddOverhang();
    
    House house_new;
    house_new.set_cad(cad);
    house_new.set_name(house.get_name());
    housegroup_new.houses.push_back(house_new);
  }  
  housegroup = housegroup_new;
}

} // DPM


