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
  reproject_factor(const Eigen::Matrix<double, 2, 1>& pIi,
				   const Eigen::Matrix<double, 3, 1>& pWi,
                      const IntrinsicParam& intrinsic,
                      const Eigen::Matrix<double, 2, 2>& sqrt_information)
      : pIi_(pIi),pWi_(pWi), intrinsic_(intrinsic),
        sqrt_information_(sqrt_information)
  {
  }

  template <typename T>
  bool operator()(const T* const p_ptr, const T* const q_ptr,
                  T* residuals_ptr) const
  {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> camera2base_p(p_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> camera2base_q(q_ptr);
	
    // Compute the relative transformation between the two frames.
    Eigen::Quaternion<T> base2camera_q = camera2base_q.conjugate();
    Eigen::Matrix<T, 3, 1> base2camera_p = base2camera_q * (-camera2base_p);
	
	Eigen::Matrix<T, 3, 1> camera_pi = base2camera_q * pWi_.template cast<T>() + base2camera_p;
	T u,v;
	u = camera_pi(0,0)/camera_pi(2,0);
	v = camera_pi(1,0)/camera_pi(2,0);
	T r = u*u+v*v;
	
	T fx(intrinsic_.fx);
	T fy(intrinsic_.fy);
	T u0(intrinsic_.u0);
	T v0(intrinsic_.u0);
	T k1(intrinsic_.u0);
	T k2(intrinsic_.u0);

	T dist = T(1.) + r*(k1 + k2*r);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_ptr);
    residuals(0, 0) =
        fx * dist * u + u0 - pIi_(0,0);
    residuals(1, 0) = 
		fy * dist * v + v0  - pIi_(1,0);

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


#endif // REPROJECT_H
