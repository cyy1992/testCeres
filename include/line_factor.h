#ifndef LINE_FACTOR_H
#define LINE_FACTOR_H

#include "Eigen/Core"
#include "ceres/autodiff_cost_function.h"

#include "types.h"
using namespace ceres::examples;

class line_factor
{
public:
	line_factor(const Pose3d& cam02world_pose,const Pose3d& world2cami_pose,
		const Eigen::Matrix<double, 1, 1>& sqrt_information,const Eigen::Matrix<double, 3, 3> &e1,const Eigen::Matrix<double, 3, 1> &e2)
		: cam02world_pose_(cam02world_pose),world2cami_pose_(world2cami_pose),sqrt_information_(sqrt_information),e1_(e1),e2_(e2)
	{
	}

	template <typename T>
	bool operator()(const T* const p_ptr, const T* const q_ptr, const T* const lineParam_ptr,T* residuals_ptr) const
	{
		
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> cam2base_p(p_ptr);
		Eigen::Map<const Eigen::Quaternion<T>> cam2base_q(q_ptr);
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> coeff(lineParam_ptr);
		
		Eigen::Matrix<T, 3, 1> base2cam_p = -(cam2base_q.conjugate()*cam2base_p);
		Eigen::Quaternion<T> world2base0_q = cam2base_q*cam02world_pose_.q.conjugate().template cast<T>();
		Eigen::Quaternion<T> world2cam0_q = cam02world_pose_.q.conjugate().template cast<T>();
		Eigen::Matrix<T, 3, 1> world2base0_p = cam2base_q*(-(world2cam0_q*cam02world_pose_.p.template cast<T>())) + cam2base_p;
		Eigen::Quaternion<T> cami2world_q = world2cami_pose_.q.template cast<T>();
		Eigen::Matrix<T, 3, 1> cami2world_p = world2cami_pose_.p.template cast<T>();
		Eigen::Matrix<T, 3, 1> basei2world_p = cami2world_q * base2cam_p + cami2world_p;
		Eigen::Matrix<T, 3, 1> basei2base0_p_cam = world2base0_q*basei2world_p + world2base0_p;
		Eigen::Matrix<T, 3, 1> re = e1_.template cast<T>()*basei2base0_p_cam + e2_.template cast<T>();
		//T residuals(residuals_ptr);
		Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(residuals_ptr);
		Eigen::Matrix<T, 3, 1> re1 = e1_.template cast<T>()*coeff.template cast<T>() + e2_.template cast<T>();
		residuals =re1.transpose()*re;
		residuals.applyOnTheLeft(sqrt_information_.template cast<T>());
		return true;
	}

	static ceres::CostFunction*
	Create(const Pose3d& cam02world_pose,const Pose3d& world2cami_pose,
		const Eigen::Matrix<double, 1, 1>& sqrt_information,const Eigen::Matrix<double, 3, 3> &e1,const Eigen::Matrix<double, 3, 1> &e2)
	{
		return new ceres::AutoDiffCostFunction<line_factor, 1, 2, 4, 3>(
			new line_factor(cam02world_pose, world2cami_pose,
									sqrt_information,e1,e2));
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // The measurement for the position of B relative to A in the A frame.
	const Pose3d cam02world_pose_;
	const Pose3d world2cami_pose_;
	const Eigen::Matrix<double, 1, 1> sqrt_information_;
	const Eigen::Matrix<double, 3, 3> e1_;
	const Eigen::Matrix<double, 3, 1> e2_;
};

class line_factor2
{
public:
	line_factor2(const Pose3d& cam02world_pose,
		const Eigen::Matrix<double, 1, 1>& sqrt_information,const Eigen::Matrix<double, 3, 3> &e1,const Eigen::Matrix<double, 3, 1> &e2)
		: cam02world_pose_(cam02world_pose),sqrt_information_(sqrt_information),e1_(e1),e2_(e2)
	{
	}

	template <typename T>
	bool operator()(const T* const cam2base_p_ptr, const T* const cam2base_q_ptr, 
				const T* const cam2world_p_ptr, const T* const cam2world_q_ptr,
				 const T* const lineParam_ptr,T* residuals_ptr) const
	{
		
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> cam2base_p(cam2base_p_ptr);
		Eigen::Map<const Eigen::Quaternion<T>> cam2base_q(cam2base_q_ptr);
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> cami2world_p(cam2world_p_ptr);
		Eigen::Map<const Eigen::Quaternion<T>> cami2world_q(cam2world_q_ptr);

		Eigen::Map<const Eigen::Matrix<T, 3, 1>> coeff(lineParam_ptr);
		
		Eigen::Matrix<T, 3, 1> base2cam_p = -(cam2base_q.conjugate()*cam2base_p);
		Eigen::Quaternion<T> world2base0_q = cam2base_q*cam02world_pose_.q.conjugate().template cast<T>();
		Eigen::Quaternion<T> world2cam0_q = cam02world_pose_.q.conjugate().template cast<T>();
		Eigen::Matrix<T, 3, 1> world2base0_p = cam2base_q*(-(world2cam0_q*cam02world_pose_.p.template cast<T>())) + cam2base_p;
// 		Eigen::Quaternion<T> cami2world_q = world2cami_pose_.q.template cast<T>();
// 		Eigen::Matrix<T, 3, 1> cami2world_p = world2cami_pose_.p.template cast<T>();
		Eigen::Matrix<T, 3, 1> basei2world_p = cami2world_q * base2cam_p + cami2world_p;
		Eigen::Matrix<T, 3, 1> basei2base0_p_cam = world2base0_q*basei2world_p + world2base0_p;
		Eigen::Matrix<T, 3, 1> re = e1_.template cast<T>()*basei2base0_p_cam + e2_.template cast<T>();
		//T residuals(residuals_ptr);
		Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(residuals_ptr);
		Eigen::Matrix<T, 3, 1> re1 = e1_.template cast<T>()*coeff.template cast<T>() + e2_.template cast<T>();
		residuals =re1.transpose()*re;
		residuals.applyOnTheLeft(sqrt_information_.template cast<T>());
		return true;
	}

	static ceres::CostFunction*
	Create(const Pose3d& cam02world_pose,
		const Eigen::Matrix<double, 1, 1>& sqrt_information,const Eigen::Matrix<double, 3, 3> &e1,const Eigen::Matrix<double, 3, 1> &e2)
	{
		return new ceres::AutoDiffCostFunction<line_factor2, 1, 2, 4,3,4, 3>(
			new line_factor2(cam02world_pose,
									sqrt_information,e1,e2));
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // The measurement for the position of B relative to A in the A frame.
	const Pose3d cam02world_pose_;
	const Eigen::Matrix<double, 1, 1> sqrt_information_;
	const Eigen::Matrix<double, 3, 3> e1_;
	const Eigen::Matrix<double, 3, 1> e2_;
};

#endif