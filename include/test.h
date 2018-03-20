#include "Eigen/Core"
#include "ceres/autodiff_cost_function.h"

#include "types.h"
using namespace ceres::examples;
class test
{
public:
  test(const double& d1)
      : d1_(d1)
  {
  }

  template <typename T>
  bool operator()(const T* const k, T* residuals_ptr) const
  {
    T theta = acos(T(d1_));
    T yaw = atan(1./tan(2.*theta));
    yaw = (*k)*yaw;
    theta = atan(1. / tan(yaw)) * 0.5;
    
    T residuals(*residuals_ptr);
    residuals =
        theta - d1_*0.5;

    // Scale the residuals by the measurement uncertainty.
    return true;
  }

  static ceres::CostFunction*
  Create(const double& d1)
  {
    return new ceres::AutoDiffCostFunction<test, 1, 1>(
        new test(d1));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // The measurement for the position of B relative to A in the A frame.
  const double d1_;
};
