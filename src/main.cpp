#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "types.h"
#include "read_g2o.h"
#include "Eigen/Core"
#include "ceres/autodiff_cost_function.h"

#include "reproject_factor.h"
#include "odom_factor.h"
using namespace std;
using namespace cv;
using ceres::AutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;


struct odomData{
	double timeStamp;
	double xPos;
	double yPos;
	double qx,qy,qz,qw;
	double tpx,tpy,tpz,tax,tay,taz;
};
namespace ceres
{
namespace examples
{
// Constructs the nonlinear least squares optimization problem from the pose
// graph constraints.
// void BuildOptimizationProblem(const MapOfPoses& camera_poses,
//                               const MapOfPoses& base_poses, Pose3d* camera2base,
//                               double* k1, double* k2,
//                               ceres::Problem* problem)
// {
//   CHECK(camera2base != NULL);
//   CHECK(camera_poses.size() == base_poses.size());
//   CHECK(problem != NULL);
//   if (camera_poses.size() < 2 || base_poses.size() < 2)
//   {
//     LOG(INFO) << "No constraints, no problem to optimize.";
//     return;
//   }
// 
//   ceres::LossFunction* loss_function = NULL;
//   //  ceres::LocalParameterization* quaternion_local_parameterization =
//   //      new EigenQuaternionParameterization;
// 
//   std::map<int, Pose3d, std::less<int>,
//            Eigen::aligned_allocator<std::pair<const int, Pose3d>>>::
//       const_iterator camera_poses_iter = camera_poses.begin();
//   std::map<int, Pose3d, std::less<int>,
//            Eigen::aligned_allocator<std::pair<const int, Pose3d>>>::
//       const_iterator base_poses_iter = base_poses.begin();
//   Pose3d prev_camera_pose = camera_poses_iter->second;
//   Pose3d prev_base_pose = base_poses_iter->second;
//   for (++camera_poses_iter, ++base_poses_iter;
//        camera_poses_iter != camera_poses.end();
//        ++camera_poses_iter, ++base_poses_iter)
//   {
//     Pose3d cur_camera_pose = camera_poses_iter->second;
//     Pose3d cur_base_pose = base_poses_iter->second;
// 
//     // calc cur2prev_camera
//     Eigen::Vector3d prev_camera_p = prev_camera_pose.p;
//     Eigen::Quaterniond prev_camera_q = prev_camera_pose.q;
//     Eigen::Vector3d cur_camera_p = cur_camera_pose.p;
//     Eigen::Quaterniond cur_camera_q = cur_camera_pose.q;
// 
//     Eigen::Quaterniond prev_camera_q_inverse = prev_camera_q.conjugate();
//     Eigen::Quaterniond cur2prev_camera_q = prev_camera_q_inverse * cur_camera_q;
//     Eigen::Vector3d cur2prev_camera_p =
//         prev_camera_q_inverse * (cur_camera_p - prev_camera_p);
//     Pose3d cur2prev_camera;
//     cur2prev_camera.p = cur2prev_camera_p;
//     cur2prev_camera.q = cur2prev_camera_q;
// 
//     // calc cur2prev_base
//     Eigen::Vector3d prev_base_p = prev_base_pose.p;
//     Eigen::Quaterniond prev_base_q = prev_base_pose.q;
//     Eigen::Vector3d cur_base_p = cur_base_pose.p;
//     Eigen::Quaterniond cur_base_q = cur_base_pose.q;
// 
//     Eigen::Quaterniond prev_base_q_inverse = prev_base_q.conjugate();
//     Eigen::Quaterniond cur2prev_base_q = prev_base_q_inverse * cur_base_q;
//     Eigen::Vector3d cur2prev_base_p =
//         prev_base_q_inverse * (cur_base_p - prev_base_p);
//     Pose3d cur2prev_base;
//     cur2prev_base.p = cur2prev_base_p;
//     cur2prev_base.q = cur2prev_base_q;
// 
//     double infomation_scale =
//         2. / (cur_camera_pose.covariance + prev_camera_pose.covariance);
// 
//     //    const Eigen::Matrix<double, 6, 6> sqrt_information =
//     //        constraint.information.llt().matrixL();
//     const Eigen::Matrix<double, 6, 6> sqrt_information =
//         Eigen::MatrixXd::Identity(6, 6) * infomation_scale;
//     // Ceres will take ownership of the pointer.
//     ceres::CostFunction* cost_function = UnionCalibErrorTerm::Create(
//         cur2prev_camera, cur2prev_base, sqrt_information);
// 
//     problem->AddResidualBlock(cost_function, loss_function,
//                               camera2base->p.data(),
//                               camera2base->q.coeffs().data(), k1, k2);
// 
//     problem->SetParameterization(camera2base->q.coeffs().data(),
//                                  g_quaternion_local_parameterization);
// 
//     prev_camera_pose = cur_camera_pose;
//     prev_base_pose = cur_base_pose;
//   }
// 
//   // The pose graph optimization problem has six DOFs that are not fully
//   // constrained. This is typically referred to as gauge freedom. You can apply
//   // a rigid body transformation to all the nodes and the optimization problem
//   // will still have the exact same cost. The Levenberg-Marquardt algorithm has
//   // internal damping which mitigates this issue, but it is better to properly
//   // constrain the gauge freedom. This can be done by setting one of the poses
//   // as constant so the optimizer cannot change it.
//   //  MapOfPoses::iterator pose_start_iter = poses->begin();
//   //  CHECK(pose_start_iter != poses->end()) << "There are no poses.";
//   //  problem->SetParameterBlockConstant(pose_start_iter->second.p.data());
//   //  problem->SetParameterBlockConstant(pose_start_iter->second.q.coeffs().data());
// }

// Returns true if the solve was successful.
bool SolveOptimizationProblem(ceres::Problem* problem)
{
  CHECK(problem != NULL);

  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.function_tolerance = 1e-8;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);

  std::cout << summary.FullReport() << '\n';

  return summary.IsSolutionUsable();
}

 //Output the poses to the file with format: id x y z q_x q_y q_z q_w.
 bool OutputPoses(const std::string& filename, const MapOfPoses& poses)
{
  std::fstream outfile;
  outfile.open(filename.c_str(), std::istream::out);
  if (!outfile)
  {
    LOG(ERROR) << "Error opening the file: " << filename;
    return false;
  }
  for (std::map<int, Pose3d, std::less<int>,
                Eigen::aligned_allocator<std::pair<const int, Pose3d>>>::
           const_iterator poses_iter = poses.begin();
       poses_iter != poses.end(); ++poses_iter)
  {
    const std::map<int, Pose3d, std::less<int>,
                   Eigen::aligned_allocator<std::pair<const int, Pose3d>>>::
        value_type& pair = *poses_iter;
    outfile << pair.first << " " << pair.second.p.transpose() << " "
            << pair.second.q.x() << " " << pair.second.q.y() << " "
            << pair.second.q.z() << " " << pair.second.q.w() << '\n';
  }
  return true;
}

} // namespace examples
} // namespace ceres

struct mPose{
	Eigen::Vector3d trans;
	Eigen::Quaterniond q;
};
void loadFiles(string fileName,vector<double> &data)
{
	ifstream fi(fileName);
	if(!fi.is_open())
	{
		cout << "can not open files!"<<endl;
		return;
	}
	
	double 	d;
	while(fi >> d)
	{
		data.push_back(d);
	}
	fi.close();
}
int main(int argc, char **argv)
{
// 	vector<odomData> odom_data;
	google::InitGoogleLogging(argv[0]);
	//CERES_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
	
	vector<double> odom_data,time_circle_data,imgPts_circle_data,worldPts_circle_data;
	string fileName_odom = "data1/odoms_circle.txt";
	string fileName_timeCircle = "data1/times_circle.txt";
	string fileName_imgPtsCircle = "data1/imgPts_circle.txt";
	string fileName_worldPtsCircle = "data1/worldPts_circle.txt";
	loadFiles(fileName_timeCircle,time_circle_data);
	loadFiles(fileName_odom,odom_data);
	loadFiles(fileName_imgPtsCircle,imgPts_circle_data);
	loadFiles(fileName_worldPtsCircle,worldPts_circle_data);
	
	IntrinsicParam intrinsic;
	intrinsic.fx = 478.654;
	intrinsic.fy = 478.618;
	intrinsic.u0 = 366.201;
	intrinsic.v0 = 208.891;
	intrinsic.k1 = -0.37251;
	intrinsic.k2 = 0.149073;
	Mat	intrinsic_ =
	(Mat_<double>(3, 3) << intrinsic.fx, 0, intrinsic.u0,
	0, intrinsic.fy, intrinsic.v0,
	0, 0, 1);
	Mat distortion_ =
		(Mat_<double>(4, 1) << intrinsic.k1,intrinsic.k2, 0, 0);
	for(unsigned int i = 0; i < time_circle_data.size()/2; i++)
	{
		double cir_time = time_circle_data[2*i];
		double ptsNum = time_circle_data[2*i+1];
		odomData tmp_odom;
		vector<Eigen::Matrix<double,2,1>> tmp_imgPts;
		vector<Eigen::Matrix<double,3,1>> tmp_worldPts;
		vector<Point3d> pWj;
		vector<Point2d> pIj;
		for(int j = 0; j < ptsNum; j++)
		{
			tmp_imgPts.push_back(Eigen::Matrix<double,2,1>(imgPts_circle_data[0],imgPts_circle_data[1]));
			pIj.push_back(Point2d(imgPts_circle_data[0],imgPts_circle_data[1]));
			for(int k = 0; k<2; k++)
				imgPts_circle_data.erase(imgPts_circle_data.begin());
			tmp_worldPts.push_back(Eigen::Matrix<double,3,1>(worldPts_circle_data[0],worldPts_circle_data[1],worldPts_circle_data[2]));
			pWj.push_back(Point3d(worldPts_circle_data[0],worldPts_circle_data[1],worldPts_circle_data[2]));
	
			for(int k = 0; k<3; k++)
				worldPts_circle_data.erase(worldPts_circle_data.begin());
		}
		ceres::Problem problem;
		ceres::LossFunction *loss_function = NULL;
		mPose mp;
		Mat tvec,rvec;
		solvePnP(pWj,pIj,intrinsic_,distortion_,rvec,tvec);
		Eigen::Vector3d trans;
		Mat_<double> Rod = Mat_<double>::ones(3,3);;
		Rodrigues(rvec,Rod);
		Eigen::Matrix<double,3,3> Rq;
		//cv2eigen(Rod,Rq);
		Rq << Rod.at<double>(0,0),Rod.at<double>(0,1),Rod.at<double>(0,2),
		Rod.at<double>(1,0),Rod.at<double>(1,1),Rod.at<double>(1,2),
		Rod.at<double>(2,0),Rod.at<double>(2,1),Rod.at<double>(2,2);
		trans << tvec.at<double>(0),tvec.at<double>(1),tvec.at<double>(2);
		double a = sqrt(rvec.at<double>(0)*rvec.at<double>(0)+ rvec.at<double>(1) *rvec.at<double>(1)+rvec.at<double>(2) *rvec.at<double>(2));
		Eigen::AngleAxisd r( a, Eigen::Vector3d ( rvec.at<double>(0),rvec.at<double>(1),rvec.at<double>(2) ) ); 
		Eigen::Quaterniond q( r );
		mp.trans = trans;
		mp.q = Eigen::Quaterniond( Rq );
		cout << setprecision (10);
		cout << mp.trans <<endl;
		double pt[6];
		pt[0] = rvec.at<double>(0);pt[1] = rvec.at<double>(1);pt[2] = rvec.at<double>(2);
		pt[3] = tvec.at<double>(0);pt[4] = tvec.at<double>(1);pt[5] = tvec.at<double>(2)+0.5;
		cout <<pt[3]<<", "<< pt[4]<<", "<<pt[5] <<endl;
		
		for(int j = 0; j < tmp_imgPts.size(); j++)
		{
			double infomation_scale =1;

			//    const Eigen::Matrix<double, 6, 6> sqrt_information =
			//        constraint.information.llt().matrixL();
			const Eigen::Matrix<double, 2, 2> sqrt_information =
				Eigen::MatrixXd::Identity(2, 2) * infomation_scale;
			// Ceres will take ownership of the pointer.
			//const Eigen::Matrix<double, 2, 1>& pIi, const Eigen::Matrix<double, 3, 1>& pWi,const IntrinsicParam& intrinsic,const Eigen::Matrix<double, 2, 2>& sqrt_information)
				
// 			ceres::CostFunction* cost_function = reproject_factor::Create(
// 				tmp_imgPts[j], tmp_worldPts[j], intrinsic,sqrt_information);
// 			
// 			problem.AddResidualBlock(cost_function, loss_function,
// 							mp.trans.data(),
// 							mp.q.coeffs().data());
			ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(
				tmp_imgPts[j], tmp_worldPts[j], intrinsic,sqrt_information);
			
			problem.AddResidualBlock(cost_function, loss_function,
							pt);
		}
		
// 		ceres::Solver::Options options;
// 		options.linear_solver_type = ceres::DENSE_SCHUR;
// 		options.minimizer_progress_to_stdout = true;
// 		ceres::Solver::Summary summary;
// 		ceres::Solve(options, &problem, &summary);
// 
		ceres::Solver::Options options;
		options.max_num_iterations = 200;
// 		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		options.linear_solver_type = ceres::DENSE_SCHUR;
		//options.function_tolerance = 1e-6;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		cout << setprecision (10);

		cout <<pt[3]<<", "<< pt[4]<<", "<<pt[5] <<endl;
		cout  <<endl;
		//std::cout << summary.FullReport() << '\n';
		

// 		while(fabs(odom_data[0] - cir_time)>0.03)
// 		{
// 			if(odom_data[0] -cir_time > 0.05)
// 			{
// 				cout << " tiao guo 66666666666666666"<<endl;
// 				break;
// 			}
// 			for(int k = 0; k < 13; k++)
// 			{
// 				odom_data.erase(odom_data.begin());
// 			}
// 			tmp_odom.timeStamp = odom_data[0];
// 			tmp_odom.xPos = odom_data[1];tmp_odom.yPos = odom_data[2];
// 			tmp_odom.qx = odom_data[3];tmp_odom.qy = odom_data[4];tmp_odom.qz = odom_data[5];tmp_odom.qw = odom_data[6];
// 			tmp_odom.tpx = odom_data[7];tmp_odom.tpy = odom_data[8];tmp_odom.tpz = odom_data[9];
// 			tmp_odom.tax = odom_data[10];tmp_odom.tay = odom_data[11];tmp_odom.taz = odom_data[12];
// 		}
		//cout <<fixed<<setprecision(10)<< cir_time <<endl;
		//cout << fixed<<setprecision(10)<<tmp_odom.timeStamp << endl;
	}
	

// 	ceres::Problem problem;
// 	for (int i = 0; i < bal_problem.num_observations(); ++i) {
// 	ceres::CostFunction* cost_function =
// 		reprojectError::Create(
// 			bal_problem.observations()[2 * i + 0],
// 			bal_problem.observations()[2 * i + 1]);
// 	problem.AddResidualBlock(cost_function,
// 							NULL /* squared loss */,
// 							bal_problem.mutable_camera_for_observation(i),
// 							bal_problem.mutable_point_for_observation(i));
// 	}
// 	ceres::Solver::Options options;
// 	options.linear_solver_type = ceres::DENSE_SCHUR;
// 	options.minimizer_progress_to_stdout = true;
// 	ceres::Solver::Summary summary;
// 	ceres::Solve(options, &problem, &summary);
// 	std::cout << summary.FullReport() << "\n";
	return 1;
}