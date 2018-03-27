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
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "types.h"
#include "read_g2o.h"

#include "reproject_factor.h"
#include "odom_factor.h"
#include "line_factor.h"
using namespace std;
using namespace cv;
using ceres::AutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
ceres::LocalParameterization* g_quaternion_local_parameterization;
IntrinsicParam intrinsic;
void buildReprojectProblem(Pose3d *cam2world_pose, vector<Eigen::Matrix<double,2,1>> imgPts,
	vector<Eigen::Matrix<double,3,1>> worldPts,ceres::Problem* problem)
{
	ceres::LossFunction* loss_function = NULL;
	double infomation_scale = cam2world_pose->covariance;
	const Eigen::Matrix<double, 2, 2> sqrt_information = Eigen::MatrixXd::Identity(2, 2) * infomation_scale;
	for(unsigned int i = 0; i< imgPts.size();i++)
	{
		Eigen::Matrix<double,2,1> imgPts_tmp = imgPts[i];
		Eigen::Matrix<double,3,1> worldPts_tmp = worldPts[i];
		ceres::CostFunction* cost_function = reproject_factor::Create(
				imgPts_tmp,worldPts_tmp, intrinsic, sqrt_information);

		problem->AddResidualBlock(cost_function, loss_function,
						cam2world_pose->p.data(),
						cam2world_pose->q.coeffs().data());

		problem->SetParameterization(cam2world_pose->q.coeffs().data(),
								g_quaternion_local_parameterization);
	}
}
void buildOdomCameraProblem(const Pose3d& base2odomPre_pose,const Pose3d& base2odomCur_pose, Pose3d* cam2worldPre_pose,Pose3d * cam2worldCur_pose,Pose3d* camera2base,
                              double* k1, double* k2,
                              ceres::Problem* problem)
{
	ceres::LossFunction* loss_function = NULL;

	// calc cur2prev_base
	Eigen::Vector3d prev_base_p = base2odomPre_pose.p;
	Eigen::Quaterniond prev_base_q = base2odomPre_pose.q;
	Eigen::Vector3d cur_base_p = base2odomCur_pose.p;
	Eigen::Quaterniond cur_base_q = base2odomCur_pose.q;

	Eigen::Quaterniond prev_base_q_inverse = prev_base_q.conjugate();
	Eigen::Quaterniond cur2prev_base_q = prev_base_q_inverse * cur_base_q;
	Eigen::Vector3d cur2prev_base_p =
		prev_base_q_inverse * (cur_base_p - prev_base_p);
	Pose3d cur2prev_base;
	cur2prev_base.p = cur2prev_base_p;
	cur2prev_base.q = cur2prev_base_q;

	double infomation_scale =
		2. / (cam2worldPre_pose->covariance + cam2worldCur_pose->covariance);

	//    const Eigen::Matrix<double, 6, 6> sqrt_information =
	//        constraint.information.llt().matrixL();
	const Eigen::Matrix<double, 6, 6> sqrt_information =
		Eigen::MatrixXd::Identity(6, 6) * infomation_scale;
	// Ceres will take ownership of the pointer.
	ceres::CostFunction* cost_function = odom_factor2::Create(
		cur2prev_base, sqrt_information);

	problem->AddResidualBlock(cost_function, loss_function,
							camera2base->p.data(),camera2base->q.coeffs().data(),
							cam2worldPre_pose->p.data(),cam2worldPre_pose->q.coeffs().data(),
							cam2worldCur_pose->p.data(),cam2worldCur_pose->q.coeffs().data(),
							k1, k2);

	problem->SetParameterization(camera2base->q.coeffs().data(),
								g_quaternion_local_parameterization);
	problem->SetParameterization(cam2worldPre_pose->q.coeffs().data(),
								g_quaternion_local_parameterization);
	problem->SetParameterization(cam2worldCur_pose->q.coeffs().data(),
								g_quaternion_local_parameterization);

}

void buildFitLineProblem(const Pose3d &cam02world_pose,Pose3d* cam2world_pose, Pose3d* camera2base, Eigen::Vector3d *coeff, ceres::Problem *problem)
{ 
	//fit line
	ceres::LossFunction* loss_function = NULL;

	vector<cv::Point2f> fit_pts_cam;
	Eigen::Matrix<double,3,3> e1;e1 << 1,0,0,0,1,0,0,0,0;
	Eigen::Matrix<double,3,1> e2; e2 << 0,0,1.0;
	double infomation_scale = 2.0/(cam2world_pose->covariance + cam02world_pose.covariance);
	const Eigen::Matrix<double, 1, 1> sqrt_information =
		Eigen::MatrixXd::Identity(1, 1) * infomation_scale;
	// Ceres will take ownership of the pointer.
	ceres::CostFunction* cost_function = line_factor2::Create(
		cam02world_pose, sqrt_information,e1,e2);

	problem->AddResidualBlock(cost_function, loss_function,
							camera2base->p.data(),camera2base->q.coeffs().data(), 
							cam2world_pose->p.data(),cam2world_pose->q.coeffs().data(), 
							coeff->data());
	
	problem->SetParameterization(camera2base->q.coeffs().data(),
								g_quaternion_local_parameterization);
	problem->SetParameterization(cam2world_pose->q.coeffs().data(),
								g_quaternion_local_parameterization);
}

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
	google::InitGoogleLogging(argv[0]);	
	vector<double> time_circle_data,imgPts_circle_data,worldPts_circle_data;
	string fileName_timeCircle = "data2/times_circle.txt";
	string fileName_imgPtsCircle = "data2/imgPts_circle.txt";
	string fileName_worldPtsCircle = "data2/worldPts_circle.txt";
	loadFiles(fileName_timeCircle,time_circle_data);
	loadFiles(fileName_imgPtsCircle,imgPts_circle_data);
	loadFiles(fileName_worldPtsCircle,worldPts_circle_data);
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
	vector<vector<Eigen::Matrix<double,2,1>>> imgPts_circle1;
	vector<vector<Eigen::Matrix<double,3,1>>> worldPts_circle1;

	for(unsigned int i = 0; i < time_circle_data.size()/2-1; i++)
	{
		//double cir_time = time_circle_data[2*i];
		double ptsNum = time_circle_data[2*i+1];
		vector<Point3d> pWj;
		vector<Point2d> pIj;
		vector<Eigen::Matrix<double,2,1>> imgPts_tmp;
		vector<Eigen::Matrix<double,3,1>> worldPts_tmp;

		for(int j = 0; j < ptsNum; j++)
		{
			imgPts_tmp.push_back(Eigen::Matrix<double,2,1>(imgPts_circle_data[0],imgPts_circle_data[1]));
			pIj.push_back(Point2d(imgPts_circle_data[0],imgPts_circle_data[1]));
			for(int k = 0; k<2; k++)
				imgPts_circle_data.erase(imgPts_circle_data.begin());
			worldPts_tmp.push_back(Eigen::Matrix<double,3,1>(worldPts_circle_data[0],worldPts_circle_data[1],worldPts_circle_data[2]));
			pWj.push_back(Point3d(worldPts_circle_data[0],worldPts_circle_data[1],worldPts_circle_data[2]));
	
			for(int k = 0; k<3; k++)
				worldPts_circle_data.erase(worldPts_circle_data.begin());
		}
		imgPts_circle1.push_back(imgPts_tmp);
		worldPts_circle1.push_back(worldPts_tmp);
	}
	
	vector<double> time_line_data,imgPts_line_data,worldPts_line_data;
	string fileName_timeLine = "data2/times_line.txt";
	string fileName_imgPtsLine = "data2/imgPts_line.txt";
	string fileName_worldPtsLine = "data2/worldPts_line.txt";
	loadFiles(fileName_timeLine,time_line_data);
	loadFiles(fileName_imgPtsLine,imgPts_line_data);
	loadFiles(fileName_worldPtsLine,worldPts_line_data);
	vector<Eigen::Quaternion<double>> w2c_vq_line; 
	vector<vector<Eigen::Matrix<double,2,1>>> imgPts_line1;
	vector<vector<Eigen::Matrix<double,3,1>>> worldPts_line1;

	vector<Vector3d> w2c_vp_line; 
	for(unsigned int i = 0; i < time_line_data.size()/2-1; i++)
	{
		//double cir_time = time_line_data[2*i];
		double ptsNum = time_line_data[2*i+1];
		vector<Point3d> pWj;
		vector<Point2d> pIj;
		vector<Eigen::Matrix<double,2,1>> imgPts_tmp;
		vector<Eigen::Matrix<double,3,1>> worldPts_tmp;
		
		for(int j = 0; j < ptsNum; j++)
		{
			imgPts_tmp.push_back(Eigen::Matrix<double,2,1>(imgPts_line_data[0],imgPts_line_data[1]));
			pIj.push_back(Point2d(imgPts_line_data[0],imgPts_line_data[1]));
			for(int k = 0; k<2; k++)
				imgPts_line_data.erase(imgPts_line_data.begin());
			worldPts_tmp.push_back(Eigen::Matrix<double,3,1>(worldPts_line_data[0],worldPts_line_data[1],worldPts_line_data[2]));
			pWj.push_back(Point3d(worldPts_line_data[0],worldPts_line_data[1],worldPts_line_data[2]));
	
			for(int k = 0; k<3; k++)
				worldPts_line_data.erase(worldPts_line_data.begin());
		}
		imgPts_line1.push_back(imgPts_tmp);
		worldPts_line1.push_back(worldPts_tmp);
	}
	
	//init 
	Eigen::Isometry3d base2camera;
	base2camera.setIdentity();
	base2camera(0, 0) = -0.01213677592071148;
	base2camera(0, 1) = -0.9999263372721495;
	base2camera(0, 2) = -0.0001367471839153443;
	base2camera(1, 0) = 0.999841336409673;
	base2camera(1, 1) = -0.01213752721085592;
	base2camera(1, 2) = 0.01303774294113711;
	base2camera(2, 0) = -0.01303843076331447;
	base2camera(2, 1) = 2.151067747224167e-005;
	base2camera(2, 2) = 0.9999149956667359;
	base2camera(0, 3) = 0.3685190219105057;
	base2camera(1, 3) = -1.875156400341508;
	base2camera(2, 3) = 82.08863295649124;

	Eigen::Quaterniond base2camera_q = Eigen::Quaterniond(base2camera.rotation());
	base2camera_q.normalize();
	
	double k1 = 1.5, k2 = 0.7;
	ceres::examples::Pose3d camera2base;
	camera2base.q = base2camera_q.conjugate();
	camera2base.p = camera2base.q * (-base2camera.translation());

	// read files
	MapOfPoses cam_line,odom_line,cam_circle,odom_circle;
	int num1,num2;
	num1 = 50;num2 = 20;
	ceres::examples::readPose3d("data2/camera_poses_line.txt",&cam_line,num2);
	ceres::examples::readPose3d("data2/base_poses_line.txt",&odom_line,num2);
	ceres::examples::readPose3d("data2/camera_poses_circle.txt",&cam_circle,num1);
	ceres::examples::readPose3d("data2/base_poses_circle.txt",&odom_circle,num1);
	
	//align data number
	vector<vector<Eigen::Matrix<double,2,1>>> imgPts_circle,imgPts_line;
	vector<vector<Eigen::Matrix<double,3,1>>> worldPts_circle,worldPts_line;

	for(unsigned int i =0;i < imgPts_circle1.size();i++)
	{
		if((i+50)%num1 == 0)
		{
			vector<Eigen::Matrix<double,2,1>> imgPts_tmp;
			vector<Eigen::Matrix<double,3,1>> worldPts_tmp;
			imgPts_tmp = imgPts_circle1[i];
			worldPts_tmp = worldPts_circle1[i];
			imgPts_circle.push_back(imgPts_tmp);
			worldPts_circle.push_back(worldPts_tmp);
		}
	}
	
	for(unsigned int i = 0; i <imgPts_line1.size();i++)
	{
		if((i+50)%num2 == 0)
		{
			vector<Eigen::Matrix<double,2,1>> imgPts_tmp;
			vector<Eigen::Matrix<double,3,1>> worldPts_tmp;
			imgPts_tmp = imgPts_line1[i];
			worldPts_tmp = worldPts_line1[i];
			imgPts_line.push_back(imgPts_tmp);
			worldPts_line.push_back(worldPts_tmp);
		}

	}
	
	Problem problem;
	g_quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;

	
	/*************************************************circle*************************************************/

	Pose3d cam_ciricle_pose[cam_circle.size()];
	Pose3d odom_circle_pose[odom_circle.size()];
	CHECK(cam_circle.size() == odom_circle.size());
	

	//start optimizing
	std::map<int, Pose3d, std::less<int>,
		Eigen::aligned_allocator<std::pair<const int, Pose3d>>>::
	const_iterator cam_circle_iter = cam_circle.begin();
	int i =0;
	for (;cam_circle_iter != cam_circle.end(); cam_circle_iter++)
	{
		cam_ciricle_pose[i] = cam_circle_iter->second;
		cam_ciricle_pose[i].p(0)+=5;
		vector<Eigen::Matrix<double,2,1>> imgPts_tmp;
		vector<Eigen::Matrix<double,3,1>> worldPts_tmp;
		imgPts_tmp = imgPts_circle[i];
		worldPts_tmp = worldPts_circle[i];
		buildReprojectProblem(&cam_ciricle_pose[i],imgPts_tmp,worldPts_tmp,&problem);
		i++;
	}
	
	//odom_factor
	std::map<int, Pose3d, std::less<int>,
			Eigen::aligned_allocator<std::pair<const int, Pose3d>>>::
		const_iterator odom_circle_iter = odom_circle.begin();
	odom_circle_pose[0] = odom_circle_iter->second;
	odom_circle_pose[0].p = 1000*odom_circle_pose[0].p;
	int j =1;
	for ( ++odom_circle_iter;  odom_circle_iter != odom_circle.end(); ++odom_circle_iter)
	{
		odom_circle_pose[j] = odom_circle_iter->second;
		odom_circle_pose[j].p = 1000 * odom_circle_pose[j].p;
		buildOdomCameraProblem(odom_circle_pose[j-1],odom_circle_pose[j],&cam_ciricle_pose[j-1],&cam_ciricle_pose[j],&camera2base,&k1,&k2,&problem);
		j++;
	}

	/*************************************************line*************************************************/
	Pose3d cam_line_pose[cam_line.size()];
	Pose3d odom_line_pose[odom_line.size()];
	CHECK(cam_line.size() == odom_line.size());
	

	//start optimizing
	std::map<int, Pose3d, std::less<int>,
		Eigen::aligned_allocator<std::pair<const int, Pose3d>>>::
	const_iterator cam_line_iter = cam_line.begin();
	int ii =0;
	for (;cam_line_iter != cam_line.end(); cam_line_iter++)
	{
		cam_line_pose[ii] = cam_line_iter->second;
		cam_line_pose[ii].p(0)+=5;
		vector<Eigen::Matrix<double,2,1>> imgPts_tmp;
		vector<Eigen::Matrix<double,3,1>> worldPts_tmp;
		imgPts_tmp = imgPts_line[ii];
		worldPts_tmp = worldPts_line[ii];
		buildReprojectProblem(&cam_line_pose[ii],imgPts_tmp,worldPts_tmp,&problem);
		ii++;
	}
	
	//odom_factor
	std::map<int, Pose3d, std::less<int>,
			Eigen::aligned_allocator<std::pair<const int, Pose3d>>>::
		const_iterator odom_line_iter = odom_line.begin();
	odom_line_pose[0] = odom_line_iter->second;
	odom_line_pose[0].p = 1000*odom_line_pose[0].p;
	int jj =1;
	Eigen::Vector3d coeff;
	coeff <<1,1,1;
	for ( ++odom_line_iter;  odom_line_iter != odom_line.end(); ++odom_line_iter)
	{
		odom_line_pose[jj] = odom_line_iter->second;
		odom_line_pose[jj].p = 1000* odom_line_pose[jj].p;
		buildOdomCameraProblem(odom_line_pose[jj-1],odom_line_pose[jj],&cam_line_pose[jj-1],&cam_line_pose[jj],&camera2base,&k1,&k2,&problem);
		buildFitLineProblem(cam_line_pose[0],&cam_line_pose[jj],&camera2base,&coeff,&problem);
		jj++;
	}

	
	
	cout << cam_ciricle_pose[0].p<<endl<<endl;
	cout <<camera2base.p <<endl<<endl;
	ceres::Solver::Options options;
	options.max_num_iterations = 200;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.function_tolerance = 1e-8;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	cout << cam_ciricle_pose[0].p<<endl<<endl;
	cout << camera2base.p <<endl<<endl;
	cout << k1<<","<<k2<<endl;


}