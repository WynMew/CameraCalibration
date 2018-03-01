#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <fstream>
#include <iostream>
#include <opencv2\calib3d\calib3d.hpp>

using namespace cv;
using namespace std;

#define LEN 5
int iFrameRow = 480;
int iFrameCol = 640;


struct FrameRepository {
	cv::Mat item_buffer[LEN];
	cv::Mat IR_buffer[LEN];
	size_t read_position;
	size_t write_position;
	std::mutex mtx;
	std::condition_variable repo_not_full;
	std::condition_variable repo_not_empty;
} gFrameRepository;


typedef struct FrameRepository FrameRepository;

VideoCapture cap;
VideoCapture IR;


bool writeMatBinary(std::ofstream& ofs, const cv::Mat& out_mat)
{
	if (!ofs.is_open()){
		return false;
	}
	if (out_mat.empty()){
		int s = 0;
		ofs.write((const char*)(&s), sizeof(int));
		return true;
	}
	int type = out_mat.type();
	ofs.write((const char*)(&out_mat.rows), sizeof(int));
	ofs.write((const char*)(&out_mat.cols), sizeof(int));
	ofs.write((const char*)(&type), sizeof(int));
	ofs.write((const char*)(out_mat.data), out_mat.elemSize() * out_mat.total());

	return true;
}


//! Save cv::Mat as binary
/*!
\param[in] filename filaname to save
\param[in] output cvmat to save
*/
bool SaveMatBinary(const std::string& filename, const cv::Mat& output){
	std::ofstream ofs(filename, std::ios::binary);
	return writeMatBinary(ofs, output);
}


//! Read cv::Mat from binary
/*!
\param[in] ifs input file stream
\param[out] in_mat mat to load
*/
bool readMatBinary(std::ifstream& ifs, cv::Mat& in_mat)
{
	if (!ifs.is_open()){
		return false;
	}

	int rows, cols, type;
	ifs.read((char*)(&rows), sizeof(int));
	if (rows == 0){
		return true;
	}
	ifs.read((char*)(&cols), sizeof(int));
	ifs.read((char*)(&type), sizeof(int));

	in_mat.release();
	in_mat.create(rows, cols, type);
	ifs.read((char*)(in_mat.data), in_mat.elemSize() * in_mat.total());

	return true;
}


//! Load cv::Mat as binary
/*!
\param[in] filename filaname to load
\param[out] output loaded cv::Mat
*/
bool LoadMatBinary(const std::string& filename, cv::Mat& output){
	std::ifstream ifs(filename, std::ios::binary);
	return readMatBinary(ifs, output);
}

void ProduceFrameItem(FrameRepository *ir, cv::Mat item, cv::Mat IRimg)
{
	unique_lock<mutex> lock(ir->mtx);
	while (((ir->write_position + 1) % LEN) == ir->read_position)
	{ // item buffer is full, just wait here.
		//std::cout << "FrameProducer is waiting for an empty slot...\n";
		(ir->repo_not_full).wait(lock); // 生产者等待"frame库缓冲区不为满"这一条件发生.
	}

	(ir->item_buffer)[ir->write_position] = item.clone();
	(ir->IR_buffer)[ir->write_position] = IRimg.clone();
	//cout << "write at " << ir->write_position << endl;
	(ir->write_position)++; // 写入位置后移.

	if (ir->write_position == LEN) // 写入位置若是在队列最后则重新设置为初始位置.
		ir->write_position = 0;

	(ir->repo_not_empty).notify_all(); // 通知消费者frame库不为空.
	lock.unlock();
}

void CameraCapture()
{
	cv::Mat img;
	cv::Mat IRimg;
	while (1)
	{
		cap >> img;
		IR >> IRimg;
		if (!img.empty() && !IRimg.empty()){
			transpose(img, img);
			transpose(IRimg, IRimg);
			//cv::cvtColor(img,img, CV_BGR2RGB);
			ProduceFrameItem(&gFrameRepository, img, IRimg);// write into frame repo
		}
	}
}

void camcalib()
{
	FrameRepository *ir = &gFrameRepository;
	cv::Mat image;
	cv::Mat IR;
	cv::Mat intrinsic = Mat(3, 3, CV_32FC1);
	cv::Mat distCoeffs;
	cv::Mat intrinsicIR = Mat(3, 3, CV_32FC1);
	cv::Mat distCoeffsIR;

	cv::Mat cameraMatrix[2];
	cv::Mat distCoeffsDual[2];

	LoadMatBinary("intrinsicB", intrinsic);
	LoadMatBinary("distCoeffsB", distCoeffs);
	LoadMatBinary("intrinsicIRB", intrinsicIR);
	LoadMatBinary("distCoeffsIRB", distCoeffsIR);


	Mat imageUndistorted;
	Mat mAlignedCam;
	Mat IRUndistorted;
	Mat mIRAlignedCam;

	int numBoards = 1;
	int numCornersHor = 7;
	int numCornersVer = 7;
	int numSquares = numCornersHor * numCornersVer;
	Size board_sz = Size(numCornersHor, numCornersVer);
	vector<vector<Point3f>> objectpoints;
	vector<vector<Point2f>> imagepoints;
	vector<vector<Point2f>> IRpoints;
	vector<Point2f> corners;
	vector<Point3f> obj;
	Mat CalibRGBImg, CalibIRImg;
	Mat grayCalibRGBImg, grayCalibIRImg;

	CalibRGBImg = imread("RGB_24.jpg");
	CalibIRImg = imread("IR_24.jpg");

	for (int j = 0; j<numSquares; j++)
		obj.push_back(Point3f(j / numCornersHor, j%numCornersHor, 0.0f));
	objectpoints.push_back(obj);

	cvtColor(CalibRGBImg, grayCalibRGBImg, CV_BGR2GRAY);
	cvtColor(CalibIRImg, grayCalibIRImg, CV_BGR2GRAY);

	bool found = findChessboardCorners(CalibRGBImg, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
	cornerSubPix(grayCalibRGBImg, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
	imagepoints.push_back(corners);

	found = findChessboardCorners(CalibIRImg, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
	cornerSubPix(grayCalibIRImg, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
	IRpoints.push_back(corners);

	//imagepoints[0].resize(1);
	//imagepoints[1].resize(1);
	//objectpoints.resize(1);

	Mat R, T, E, F;


	image = CalibRGBImg.clone();
	IR = CalibIRImg.clone();
	undistort(image, imageUndistorted, intrinsic, distCoeffs);
	undistort(IR, IRUndistorted, intrinsicIR, distCoeffsIR);
	hconcat(image, imageUndistorted, mAlignedCam);
	imshow("RGB calib", mAlignedCam);
	hconcat(IR, IRUndistorted, mIRAlignedCam);
	imshow("IR calib", mIRAlignedCam);
	waitKey(1);

	double rms = stereoCalibrate(objectpoints, imagepoints, IRpoints,
			intrinsic, distCoeffs,
			intrinsicIR, distCoeffsIR,
			image.size(), R, T, E, F,
			TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 50, 1e-6),
			CV_CALIB_FIX_INTRINSIC +
			CV_CALIB_USE_INTRINSIC_GUESS
			);

	ofstream RFile, TFile, EFile, FFile;
	RFile.open("RFile.txt");
	RFile << R;
	RFile.close();
	TFile.open("TFile.txt");
	TFile << T;
	TFile.close();
	EFile.open("EFile.txt");
	EFile << E;
	EFile.close();
	FFile.open("FFile.txt");
	FFile << F;
	FFile.close();

}

void InitFrameRepository(FrameRepository *ir)
{
	ir->write_position = 0;
	ir->read_position = 0;
	cv::Mat matrix = cv::Mat(iFrameCol, iFrameRow, CV_32F, cv::Scalar::all(0));// transposed img size
	for (int i = 0; i < LEN; i++)
	{
		ir->item_buffer[i] = matrix.clone(); // write in zeros
		ir->IR_buffer[i] = matrix.clone();
	}
}


int main()
{
	cap = VideoCapture(1);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, iFrameCol);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, iFrameRow);
	cap.set(CV_CAP_PROP_AUTO_EXPOSURE, 1);

	IR = VideoCapture(0);
	IR.set(CV_CAP_PROP_FRAME_WIDTH, iFrameCol);
	IR.set(CV_CAP_PROP_FRAME_HEIGHT, iFrameRow);
	//cap.set(CV_CAP_PROP_EXPOSURE, -3); 
	IR.set(CV_CAP_PROP_AUTO_EXPOSURE, 1);

	InitFrameRepository(&gFrameRepository);
	std::thread CameraCaptureTask(CameraCapture);
	std::thread CameraCalibTask(camcalib);

	CameraCaptureTask.join();
	CameraCalibTask.join();

	return 0;
}
