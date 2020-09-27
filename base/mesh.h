#ifndef BASE_MESH_H_
#define BASE_MESH_H_

#include <Eigen/Dense>
#include <string>
#include <vector>


namespace DPM {

struct Mesh {
  std::vector<Eigen::Vector3d> vertices;
  std::vector<Eigen::Vector3i> faces;
  bool Read(const std::string& filename);

  bool Write(const std::string& filename) const;
  bool Merge(const Mesh& mesh);
  friend std::ostream& operator<<(std::ostream& ostr, const Mesh& lhs);
};

struct MeshGroup {
  std::vector<Mesh> meshes;
  bool Read(const std::string& filename);
  bool Write(const std::string& filename) const;

  bool ReadData(const std::string& filename);
  bool WriteData(const std::string& filename) const;
};


} // DPM

#endif  // BASE_MESH_H_