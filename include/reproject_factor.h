#ifndef REPROJECT_H
#define REPROJECT_H
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <math.h>
#include "utility.h"
#include "types.h"
using ceres::AutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using namespace Eigen;  
using namespace ceres::examples;

class reproject_factor
{
public:
  reproject_factor(const Eigen::Matrix<double, 2, 1> pIi,
				   const Eigen::Matrix<double, 3, 1> pWi,
                      const IntrinsicParam intrinsic,
                      const Eigen::Matrix<double, 2, 2> sqrt_information)
      : pIi_(pIi),pWi_(pWi), intrinsic_(intrinsic),
        sqrt_information_(sqrt_information)
  {
  }

  template <typename T>
  bool operator()(const T* const p_ptr, const T* const q_ptr,
                  T* residuals_ptr) const
  {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> cam2world_p(p_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> cam2world_q(q_ptr);
	
    // Compute the relative transformation between the two frames.
//     Eigen::Quaternion<T> base2camera_q = camera2base_q.conjugate();
//     Eigen::Matrix<T, 3, 1> base2camera_p = base2camera_q * (-camera2base_p);
	
	Eigen::Matrix<T, 3, 1> camera_pi = cam2world_q.conjugate() *( pWi_.template cast<T>()-  cam2world_p);
	T u,v;
	u = camera_pi(0,0)/camera_pi(2,0);
	v = camera_pi(1,0)/camera_pi(2,0);
	T r = u*u+v*v;
// 	
	const T& fx = T(intrinsic_.fx);
	const T& fy = T(intrinsic_.fy);
	const T& u0 = T(intrinsic_.u0);
	const T& v0 = T(intrinsic_.v0);
	T k1(intrinsic_.k1);
	T k2(intrinsic_.k2);

	T dist = 1.0 + r*(k1 + k2*r);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_ptr);
    T predicted_x = fx  * dist * u  + u0;
	T predicted_y = fy  * dist * v   + v0; 

	residuals(0, 0) = predicted_x - T(pIi_(0,0));
	residuals(1, 0) = predicted_y - T(pIi_(1,0));

    // Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());
	return true;
  }

  static ceres::CostFunction*
  Create(const Eigen::Matrix<double, 2, 1>& pIi, const Eigen::Matrix<double, 3, 1>& pWi,const IntrinsicParam& intrinsic,
         const Eigen::Matrix<double, 2, 2>& sqrt_information)
  {
    return new ceres::AutoDiffCostFunction<reproject_factor, 2, 3, 4>(
        new reproject_factor(pIi,pWi, intrinsic,
                                sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  const Eigen::Matrix<double, 2, 1> pIi_;
  const Eigen::Matrix<double, 3, 1> pWi_;
  const IntrinsicParam intrinsic_;
  const Eigen::Matrix<double, 2, 2> sqrt_information_;
};


struct SnavelyReprojectionError {
  SnavelyReprojectionError(const Eigen::Matrix<double, 2, 1> pIi,
				   const Eigen::Matrix<double, 3, 1> pWi,
                      const IntrinsicParam intrinsic,
                      const Eigen::Matrix<double, 2, 2> sqrt_information)
      : pIi_(pIi),pWi_(pWi), intrinsic_(intrinsic),
        sqrt_information_(sqrt_information){}

  template <typename T>
  bool operator()(const T* const camera,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
	T point[3];
	point[0] = T(pWi_(0,0));point[1] = T(pWi_(1,0));point[2] = T(pWi_(2,0));
    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp =  p[0] / p[2];
    T yp =  p[1] / p[2];

    // Apply second and fourth order radial distortion.
//     const T& l1 = camera[7];
//     const T& l2 = camera[8];
	const T fx(intrinsic_.fx);
	const T fy(intrinsic_.fy);
	const T u0(intrinsic_.u0);
	const T v0(intrinsic_.v0);
	const T k1(intrinsic_.k1);
	const T k2(intrinsic_.k2);
    T r2 = xp*xp + yp*yp;
    T distortion = 1.0 + r2  * (k1 + k2  * r2);

    // Compute final projected point position.
    T predicted_x = fx * distortion * xp + u0;
    T predicted_y = fy * distortion * yp + v0;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - pIi_(0,0);
    residuals[1] = predicted_y - pIi_(1,0);

    return true;
  }
    static ceres::CostFunction* Create(const Eigen::Matrix<double, 2, 1>& pIi,
				   const Eigen::Matrix<double, 3, 1>& pWi,
                      const IntrinsicParam& intrinsic,
                      const Eigen::Matrix<double, 2, 2>& sqrt_information) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6>(
                new SnavelyReprojectionError(pIi,pWi, intrinsic,
                                sqrt_information)));
  }

  const Eigen::Matrix<double, 2, 1> pIi_;
  const Eigen::Matrix<double, 3, 1> pWi_;
  const IntrinsicParam intrinsic_;
  const Eigen::Matrix<double, 2, 2> sqrt_information_;
};


#endif // REPROJECT_H
