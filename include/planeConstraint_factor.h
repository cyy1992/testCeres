#ifndef PLANE_CONSTRAINT_FACOTR_H
#define PLANE_CONSTRAINT_FACOTR_H

#include "Eigen/Core"
#include "ceres/autodiff_cost_function.h"

#include "types.h"
using namespace ceres::examples;

class planeConstraint_factor{
public:
	planeConstraint_factor(const Pose3d& cur2prev_camera,
                      const Pose3d& cur2prev_base,
                      const Eigen::Matrix<double, 6, 6>& sqrt_information)
      : cur2prev_camera_(cur2prev_camera), cur2prev_base_(cur2prev_base),
        sqrt_information_(sqrt_information)
	{
	}

	template <typename T>
	bool operator()(const T* const p_ptr, const T* const q_ptr, const T* k1_ptr,
					const T* k2_ptr, T* residuals_ptr) const
	{
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> camera2base_p(p_ptr);
		Eigen::Map<const Eigen::Quaternion<T>> camera2base_q(q_ptr);

		// Compute the relative transformation between the two frames.
		Eigen::Quaternion<T> base2camera_q = camera2base_q.conjugate();
		Eigen::Matrix<T, 3, 1> base2camera_p = base2camera_q * (-camera2base_p);

		Eigen::Quaternion<T> curBase2preCamera_q =
			cur2prev_camera_.q.template cast<T>() * base2camera_q;
		Eigen::Matrix<T, 3, 1> curBase2preCamera_p =
			cur2prev_camera_.q.template cast<T>() * base2camera_p +
			cur2prev_camera_.p.template cast<T>();

		Eigen::Quaternion<T> curBase2preBase_q =
			camera2base_q * curBase2preCamera_q;
		Eigen::Matrix<T, 3, 1> curBase2preBase_p =
			camera2base_q * curBase2preCamera_p + camera2base_p;
		
		T qw = cur2prev_base_.q.template cast<T>().w();
		T theta = acos(qw);
		T yaw = atan(1./tan(2.*theta));
		yaw = (*k2_ptr)*yaw;
		theta = atan(1. / tan(yaw)) * 0.5;
		Eigen::Quaternion<T> cur2prev_base_q_scaled(cos(theta), T(0.), T(0.), sin(theta));
		Eigen::Matrix<T, 3, 1> cur2prev_base_p_scaled =
			(*k1_ptr) * cur2prev_base_.p.template cast<T>();

		// Compute the error between the two orientation estimates.
		Eigen::Quaternion<T> delta_q =
			cur2prev_base_q_scaled * curBase2preBase_q.conjugate();

		// Compute the residuals.
		// [ position         ]   [ delta_p          ]
		// [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
		Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
		residuals.template block<3, 1>(0, 0) =
			curBase2preBase_p - cur2prev_base_p_scaled;
		residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

		// Scale the residuals by the measurement uncertainty.
		residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

		return true;
	}

	static ceres::CostFunction*
	Create(const Pose3d& cur2prev_camera, const Pose3d& cur2prev_base,
			const Eigen::Matrix<double, 6, 6>& sqrt_information)
	{
		return new ceres::AutoDiffCostFunction<planeConstraint_factor, pointNum_*3, 2, 4, 1, 1>(
			new planeConstraint_factor(cur2prev_camera, cur2prev_base,
									sqrt_information));
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
	// The measurement for the position of B relative to A in the A frame.
	const Pose3d cur2prev_camera_;
	const Pose3d cur2prev_base_;
	// The square root of the measurement information matrix.
	const Eigen::Matrix<double, 6, 6> sqrt_information_;
	
	const int pointNum_;

};

#endif