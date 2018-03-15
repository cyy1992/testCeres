// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2016 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: vitus@google.com (Michael Vitus)

#ifndef EXAMPLES_CERES_TYPES_H_
#define EXAMPLES_CERES_TYPES_H_

#include <istream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace ceres {
namespace examples {

struct Pose3d {
  Eigen::Vector3d p;
  Eigen::Quaterniond q;
  double covariance;

  // The name of the data type in the g2o file format.
  static std::string name() {
    return "VERTEX_SE3:QUAT";
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
struct IntrinsicParam {
	double fx,fy,u0,v0,k1,k2;
};
std::istream& operator>>(std::istream& input, Pose3d& pose) {
  input >> pose.p.x() >> pose.p.y() >> pose.p.z() >> pose.q.x() >>
      pose.q.y() >> pose.q.z() >> pose.q.w() >> pose.covariance;
  // Normalize the quaternion to account for precision loss due to
  // serialization.
  pose.q.normalize();
  return input;
}

typedef std::map<int, Pose3d, std::less<int>,
                 Eigen::aligned_allocator<std::pair<const int, Pose3d> > >
    MapOfPoses;

bool readPose3d(const std::string& filename, MapOfPoses* poses)
{
  assert(poses != NULL);

  poses->clear();

  std::ifstream infile(filename.c_str());
  if (!infile) {
    return false;
  }

  int id = 0;
  while (infile.good()) {
    id++;

    Pose3d pose;
    infile >> pose;
    // Clear any trailing whitespace from the line.
    infile >> std::ws;

    if((id+50)%100 != 0)
      continue;

    (*poses)[id] = pose;
  }

  infile.close();
  return true;
}

}  // namespace examples
}  // namespace ceres

#endif  // EXAMPLES_CERES_TYPES_H_