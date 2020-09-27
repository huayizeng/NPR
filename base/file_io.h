#ifndef FILE_IO_H__
#define FILE_IO_H__

#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <sys/stat.h>

namespace {

static bool exist_and_mkdir(const char *dirName) {
  std::ifstream infile(dirName);
  if(!infile.good()) {
    const int dir_err = mkdir(dirName, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (-1 == dir_err) {
      printf("Error creating directory\n");
      exit(1);
    }
  }
}

}

namespace DPM {

class FileIO {
public:
  FileIO(const std::string data_directory, const std::string exp) : data_directory_(data_directory), exp_(exp) {}

  // Input directory
  std::string GetDSM(const std::string& prefix) const {
    sprintf(buffer_, "%s/input/%s.h5", data_directory_.c_str(), prefix.c_str());
    return buffer_;
  }
  std::string GetDTM(const std::string& prefix) const {
    sprintf(buffer_, "%s/input/%s.h5", data_directory_.c_str(), prefix.c_str());
    return buffer_;
  }
  std::string GetDSMRoot() const {
    sprintf(buffer_, "%s/input/dsm", data_directory_.c_str());
    return buffer_;
  }
  std::string GetDTMRoot() const {
    sprintf(buffer_, "%s/input/dtm", data_directory_.c_str());
    return buffer_;
  }
  std::string GetDiffRoot() const {
    sprintf(buffer_, "%s/input/diff/%s", data_directory_.c_str(), exp_.c_str());
    return buffer_;
  }
  std::string GetOffsetDir() const {
    sprintf(buffer_, "%s/input/offset/%s", data_directory_.c_str(), exp_.c_str());
    return buffer_;
  }
  std::string GetDNNFeatures() const {
    sprintf(buffer_, "%s/features_dnn/%s", data_directory_.c_str(), exp_.c_str());
    return buffer_;    
  }
  std::string GetCadSavedDir() const { // used for add overhang
    sprintf(buffer_, "%s/output/cad_saved", data_directory_.c_str());
    return buffer_;
  }


  // Output directory
  std::string GetDepthImgDir() const {
    sprintf(buffer_, "%s/output/depthimg/%s", data_directory_.c_str(), exp_.c_str());
    exist_and_mkdir(buffer_);
    return buffer_;
  }
  std::string GetNormalImgDir() const {
    sprintf(buffer_, "%s/output/normalimg/%s", data_directory_.c_str(), exp_.c_str());
    exist_and_mkdir(buffer_);
    return buffer_;
  }
  std::string GetCadDir() const {
    sprintf(buffer_, "%s/output/cad/%s", data_directory_.c_str(), exp_.c_str());
    exist_and_mkdir(buffer_);
    return buffer_;
  }
  std::string GetCadOptimizedDir() const {
    sprintf(buffer_, "%s/output/cad_opt/%s", data_directory_.c_str(), exp_.c_str());
    exist_and_mkdir(buffer_);
    return buffer_;
  }
  std::string GetCadAddOverhangDir() const {
    sprintf(buffer_, "%s/output/cad_add_overhang/%s", data_directory_.c_str(), exp_.c_str());
    exist_and_mkdir(buffer_);
    return buffer_;
  }
  std::string GetPointCloudDir() const {
    sprintf(buffer_, "%s/output/pointcloud/%s", data_directory_.c_str(), exp_.c_str());
    exist_and_mkdir(buffer_);
    return buffer_;
  }
  std::string GetXYZNDir() const {
    sprintf(buffer_, "%s/output/xyzn/%s", data_directory_.c_str(), exp_.c_str());
    exist_and_mkdir(buffer_);
    return buffer_;
  }
  std::string GetStatesDir() const {
    sprintf(buffer_, "%s/output/states/%s", data_directory_.c_str(), exp_.c_str());
    exist_and_mkdir(buffer_);
    return buffer_;
  }


  std::string GetPolygonModelDir() const {
    sprintf(buffer_, "%s/output/ploygon_model", data_directory_.c_str());
    return buffer_;
  }
  std::string GetDepthImgSyntheticDir() const {
    sprintf(buffer_, "%s/synthetic/depthimg", data_directory_.c_str());
    return buffer_;
  }
  std::string GetSyntheticDormerHeatmapDir() const {
    sprintf(buffer_, "%s/synthetic/heatmap", data_directory_.c_str());
    return buffer_;
  }
  std::string GetNormalImgSyntheticDir() const {
    sprintf(buffer_, "%s/synthetic/normalimg", data_directory_.c_str());
    return buffer_;
  }
  std::string GetCadSyntheticDir() const {
    sprintf(buffer_, "%s/synthetic/cad", data_directory_.c_str());
    return buffer_;
  }
  std::string GetPointCloudSyntheticDir() const {
    sprintf(buffer_, "%s/synthetic/pointcloud", data_directory_.c_str());
    return buffer_;
  }
  std::string GetAnnoVerticesDir() const {
    sprintf(buffer_, "%s/anno/vertices", data_directory_.c_str());
    return buffer_;
  }

private:
  const std::string data_directory_;
  const std::string exp_;
  mutable char buffer_[1024];
};

} // DPM

#endif  // FILE_IO_H__
