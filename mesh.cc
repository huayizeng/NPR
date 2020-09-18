#include "mesh.h"

#include <iostream>
#include <limits>
#include <fstream>
#include <tuple>
#include <map>

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector3i;

using std::cerr;
using std::cout;
using std::endl;
using std::flush;
using std::ifstream;
using std::istringstream;
using std::max;
using std::min;
using std::numeric_limits;
using std::ofstream;
using std::string;


namespace {

inline std::tuple<std::vector<Eigen::Vector3d>, std::vector<std::size_t>, std::vector<std::size_t> > unique_idx_inv(const std::vector<Eigen::Vector3d> &a) {
  auto comp = [](const Eigen::Vector3d&a, const Eigen::Vector3d& b) { return a[0] * 100 + a[1] * 10 + a[2] < b[0] * 100 + b[1] * 10 + b[2]; };
  std::size_t               ind;
  std::map<Eigen::Vector3d, std::size_t, decltype(comp)>  m(comp);
  std::vector<Eigen::Vector3d>            uniques;
  std::vector<std::size_t>  idx;
  std::vector<std::size_t>  inv;
  inv.reserve(a.size());
  ind = 0U;
  for ( std::size_t i = 0U ; i < a.size() ; ++i ) {
    auto e = m.insert(std::make_pair(a[i], ind));
    if ( e.second ) {
      uniques.push_back(a[i]);
      idx.push_back(i);
      ++ind;
    }
    inv.push_back(e.first->second);
  }
  return std::make_tuple(uniques, idx, inv);
}


void UniqueIndex(DPM::Mesh& mesh) {
  DPM::Mesh mesh_copy = mesh;
  auto tup = unique_idx_inv(mesh_copy.vertices);
  mesh_copy.vertices = std::get<0>(tup);
  auto inv_inds = std::get<2>(tup);
  for (auto && face : mesh_copy.faces)
    for (unsigned i = 0; i < 3; ++i)
      face[i] = inv_inds[face[i]];
  mesh = mesh_copy;
}


void AdjustIndex(const DPM::Mesh& mesh_org, DPM::Mesh& mesh) {
  std::map<int, int> old_to_new;
  int new_index = 0;
  for (const auto& face : mesh.faces) {
    for (int i = 0; i < 3; ++i) {
      const int index = face[i];
      if (old_to_new.find(index) != old_to_new.end())
        continue;
      old_to_new[index] = new_index++;
      mesh.vertices.push_back(mesh_org.vertices[index]);
    }
  }

  for (auto& face : mesh.faces) {
    for (int i = 0; i < 3; ++i) {
      face[i] = old_to_new[face[i]];
    }
  }
}

}  // empty namespace

namespace DPM {

bool Mesh::Merge(const Mesh& mesh) {
  // Add vertices.
  const int offset = vertices.size();
  vertices.insert(vertices.end(), mesh.vertices.begin(), mesh.vertices.end());

  for (auto face : mesh.faces) {
    for (int i = 0; i < 3; ++i)
      face[i] += offset;

    faces.push_back(face);
  }
  return true;
}

bool Mesh::Read(const std::string& filename) {
  *this = Mesh();

  ifstream ifstr;
  ifstr.open(filename.c_str());
  if (!ifstr.is_open()) {
    cerr << "Cannot open a file: " << filename << endl;
    return false;
  }

  std::string line;
  while (getline(ifstr, line)) {
    if (line == "")
      continue;
    istringstream isstr(line);
    std::string first_word;
    isstr >> first_word;
    if (first_word == ""       || first_word == "#" ||
        first_word == "mtllib" || first_word == "usemtl" ||
        first_word == "vt"     || first_word == "vn" ||
        first_word == "g"      || first_word == "s") {
      continue;
    } else if (first_word == "v") {
      Vector3d vertex;
      for (int i = 0; i < 3; ++i)
        isstr >> vertex[i];
      vertices.push_back(vertex);
    } else if (first_word == "f") {
      Vector3i face;
      for (int i = 0; i < 3; ++i) {
        isstr >> face[i];
        --face[i];
      }
      faces.push_back(face);
    } else {
      // cout << "Unprocessed line: " << first_word << endl;
      cout << '.';
    }
  }
  cout << flush;
  ifstr.close();

  return true;
}

bool Mesh::Write(const std::string& filename) const {

  // std::cout << " good()=" << ofstr.good();
  // std::cout << " eof()=" << ofstr.eof();
  // std::cout << " fail()=" << ofstr.fail();
  // std::cout << " bad()=" << ofstr.bad();
  
  // cerr << "filename: filename: " << filename << endl;
  ofstream ofstr(filename.c_str());
  if (!ofstr.good()){
    cerr << "Cannot open a file: " << filename << endl;
    return false;
  }
  for (const auto& vertex : vertices)
    ofstr << "v " << vertex.transpose() << endl;
  for (const auto& face : faces)
    ofstr << "f "
          << face[0] + 1 << ' '
          << face[1] + 1 << ' '
          << face[2] + 1 << endl;
  ofstr.close();
  return true;
}

std::ostream& operator<<(std::ostream& ostr, const Mesh& lhs) {
  ostr << "Mesh" << std::endl;
  ostr << "Mesh.faces" << std::endl;
  for (auto& face : lhs.faces)
    ostr << face.transpose() << endl;
  ostr << "Mesh.vertices" << std::endl;
  for (auto& vertex : lhs.vertices)
    ostr << vertex.transpose() << endl;
  return ostr;
};

bool MeshGroup::Read(const std::string& filename) {
  *this = MeshGroup();

  Mesh mesh_org;
  if (!mesh_org.Read(filename))
    return false;

  // Find groups.
  ifstream ifstr;
  ifstr.open(filename.c_str());
  if (!ifstr.is_open()) {
    cerr << "Cannot open a file: " << filename << endl;
    return false;
  }

  std::string line;
  Mesh mesh;
  while (getline(ifstr, line)) {
    if (line == "")
      continue;
    istringstream isstr(line);
    std::string first_word;
    isstr >> first_word;

    if (first_word == "g") {
      if (!mesh.faces.empty()) {
        AdjustIndex(mesh_org, mesh);
        meshes.push_back(mesh);
        mesh = Mesh();
      }
    } else if (first_word == ""       || first_word == "#" ||
               first_word == "mtllib" || first_word == "usemtl" ||
               first_word == "vt"     || first_word == "vn" ||
               first_word == "s"      || first_word == "v") {
      continue;
    } else if (first_word == "f") {
      Vector3i face;
      for (int i = 0; i < 3; ++i) {
        isstr >> face[i];
        --face[i];
      }
      mesh.faces.push_back(face);
    } else {
      // cout << "Unprocessed line: " << first_word << endl;
      cout << '.';
    }
  }
  if (!mesh.faces.empty()) {
    AdjustIndex(mesh_org, mesh);
    meshes.push_back(mesh);
  }
  cout << flush;

  ifstr.close();
  return true;
}

bool MeshGroup::Write(const std::string& filename) const {
  ofstream ofstr(filename);
  if (!ofstr.is_open())
    return false;

  int vertex_offset = 0;
  int i = 0;
  int vertex_offset2 = 0;
  for (const auto& mesh : meshes) {
    // const int red = rand() % 255;
    // const int green = rand() % 255;
    // const int blue = rand() % 255;
    const int red = 0;
    const int green = 127;
    const int blue = 255;

    for (const auto& vertex : mesh.vertices)
      ofstr << "v " << vertex.transpose() << ' '
            << red << ' ' << green << ' ' << blue << endl;

    for (const auto& face : mesh.faces) {
      ofstr << "f "
            << face[0] + vertex_offset + 1 << ' '
            << face[1] + vertex_offset + 1 << ' '
            << face[2] + vertex_offset + 1 << endl;
    }
    vertex_offset += mesh.vertices.size();
    i++;
  }
  ofstr.close();
  return true;
}

bool MeshGroup::ReadData(const std::string& filename) {
  ifstream ifstr;
  ifstr.open(filename);
  if (!ifstr.is_open())
    return false;

  std::string stmp;
  int num_meshes;
  ifstr >> stmp >> num_meshes;
  meshes.resize(num_meshes);
  for (int i = 0; i < num_meshes; ++i) {
    int num_vertices;
    ifstr >> num_vertices;
    meshes[i].vertices.resize(num_vertices);
    for (int v = 0; v < num_vertices; ++v) {
      for (int j = 0; j < 3; ++j) {
        ifstr >> meshes[i].vertices[v][j];
      }
    }
    int num_faces;
    ifstr >> num_faces;
    meshes[i].faces.resize(num_faces);
    for (int f = 0; f < num_faces; ++f) {
      for (int j = 0; j < 3; ++j) {
        ifstr >> meshes[i].faces[f][j];
      }
    }
    UniqueIndex(meshes[i]);
  }

  return true;
}

bool MeshGroup::WriteData(const std::string& filename) const {
  ofstream ofstr;
  ofstr.open(filename);
  if (!ofstr.is_open())
    return false;

  ofstr << "MeshGroup" << endl
        << meshes.size() << endl;
  int i = 0;
  for (const auto& mesh : meshes) {
    ofstr << mesh.vertices.size() << endl;
    for (const auto& vertex : mesh.vertices) {
      ofstr << vertex.transpose() << endl;
    }
    ofstr << mesh.faces.size() << endl;
    for (const auto& face : mesh.faces) {
      ofstr << face.transpose() << endl;
    }
    i++;
  }

  return true;
}

} // DPM