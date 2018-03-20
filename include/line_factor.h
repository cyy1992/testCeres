#ifndef LINE_FACTOR_H
#define LINE_FACTOR_H

#include "Eigen/Core"
#include "ceres/autodiff_cost_function.h"

#include "types.h"
using namespace ceres::examples;
class odom_factor
{
public:
	odom_factor(const Matrix<double,3,1>& world2cam_trans,
		const double& sqrt_information)
		: world2cam_trans_(world2cam_trans),sqrt_information_(sqrt_information)
	{
	}

	template <typename T>
	bool operator()(const T* const p_ptr, const T* const q_ptr, const T* const lineParam_ptr,T* residuals_ptr) const
	{
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> camera2base_p(p_ptr);
		Eigen::Map<const Eigen::Quaternion<T>> camera2base_q(q_ptr);
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> w2c_p(world2cam_trans_);
		
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> w2b_p = camera2base_q*w2c_p + camera2base_p;
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> line_param(lineParam_ptr);

		T residuals(residuals_ptr);
		residuals = line_param.transpose()*w2b_p;
		
		residuals.applyOnTheLeft(sqrt_information_);

		return true;
	}

	static ceres::CostFunction*
	Create(const Pose3d& cur2prev_camera, const Pose3d& cur2prev_base,
			const double& sqrt_information)
	{
		return new ceres::AutoDiffCostFunction<odom_factor, 1, 2, 4, 3>(
			new odom_factor(cur2prev_camera, cur2prev_base,
									sqrt_information));
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // The measurement for the position of B relative to A in the A frame.
	const Eigen::Matrix<double,3,1> world2cam_trans_;
	const double sqrt_information_;
	cur2prev_base_p_scaled curBase2preBase_p;
};


#endif