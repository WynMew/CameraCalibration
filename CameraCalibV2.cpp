//#include <opencv2\highgui\highgui.hpp>
//#include <opencv2\imgproc\imgproc.hpp>
//#include "VideoFaceDetector.h"
//#include <sstream>
//#include <cv.h>
//#include <highgui.h>
//#include <iostream>
//#include <fstream>
//
//
//using namespace cv;
//using namespace std;

//void main(int argc, char** argv)
//{
//	int RGBVideoCap = 1;
//	int IRVideoCap = 0;
//
//	VideoFaceDetector *VFD = new VideoFaceDetector(RGBVideoCap, IRVideoCap);
//	std::thread VideoCaptureTask = VFD->VideoCaptureThread();
//	std::thread FaceProposeTask = VFD->FaceProposerThread();
//	std::thread LocalVerifyTask = VFD->LocalVerifierThread();
//
//	VideoCaptureTask.join();
//	FaceProposeTask.join();
//	LocalVerifyTask.join();
//	delete VFD;
//}


//#include <opencv2\highgui\highgui.hpp>
//#include <opencv2\imgproc\imgproc.hpp>
//#include "VideoFaceDetector.h"
//#include <sstream>
//#include <cv.h>
//#include <highgui.h>
//#include <iostream>
//#include <fstream>
//
//
//using namespace cv;
//using namespace std;
//
//
//void main(int argc, char** argv)
//{
//	int RGBVideoCap = 1;
//	int IRVideoCap = 0;
//	int iFrameRow;
//	int iFrameCol;
//	iFrameRow = 480;
//	iFrameCol = 640;
//
//
//	VideoCapture cap;
//	VideoCapture IR;
//
//	cap = VideoCapture(RGBVideoCap);
//	IR = VideoCapture(IRVideoCap);
//	cap.set(CV_CAP_PROP_FRAME_WIDTH, iFrameCol);
//	cap.set(CV_CAP_PROP_FRAME_HEIGHT, iFrameRow);
//	cap.set(CV_CAP_PROP_AUTO_EXPOSURE, 1);
//	IR.set(CV_CAP_PROP_FRAME_WIDTH, iFrameCol);
//	IR.set(CV_CAP_PROP_FRAME_HEIGHT, iFrameRow);
//	IR.set(CV_CAP_PROP_AUTO_EXPOSURE, 1);
//
//
//	cv::Mat RGBimg;
//	cv::Mat IRimg;
//	cv::Mat mFrameNow;
//	int counter = 0;
//	while (1)
//	{
//		cap >> RGBimg;
//		IR >> IRimg;
//
//		if (!RGBimg.empty() && !IRimg.empty())
//		{
//			transpose(RGBimg, RGBimg);
//			transpose(IRimg, IRimg);
//			hconcat(RGBimg, IRimg, mFrameNow);
//			imshow("Frame Now", mFrameNow);
//			cvWaitKey(1);
//
//			std::ostringstream RGBfileNameSS;
//			RGBfileNameSS <<"RGB_"<< counter << ".jpg";
//			string RGBFileName = RGBfileNameSS.str();
//
//			std::ostringstream IRfileNameSS;
//			IRfileNameSS<<"IR_" << counter << ".jpg";
//			string IRFileName = IRfileNameSS.str();
//			
//			imwrite(RGBFileName, RGBimg);
//			imwrite(IRFileName, IRimg);
//
//			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
//			counter++;
//		}
//
//	}
//
//
//}


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
	//int numBoards = 1;
	//int numCornersHor = 7;
	//int numCornersVer = 7;

	//int numSquares = numCornersHor * numCornersVer;
	//Size board_sz = Size(numCornersHor, numCornersVer);
	VideoCapture capture = VideoCapture(0);

	//vector<vector<Point3f>> object_points;
	//vector<vector<Point2f>> image_points;

	//vector<Point2f> corners;
	//int successes = 0;

	FrameRepository *ir = &gFrameRepository;
	Mat image;
	//Mat gray_image;

	//image = imread("RGB_24.jpg");

	//vector<Point3f> obj;
	//for (int j = 0; j<numSquares; j++)
	//	obj.push_back(Point3f(j / numCornersHor, j%numCornersHor, 0.0f));

	//cvtColor(image, gray_image, CV_BGR2GRAY);
	//bool found = findChessboardCorners(image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
	//if (found)
	//	{
	//		cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
	//		drawChessboardCorners(gray_image, board_sz, corners, found);
	//	}

	//imshow("grey calibration input img", gray_image);
	//cvWaitKey(1);
	//image_points.push_back(corners);
	//object_points.push_back(obj);

	Mat intrinsic = Mat(3, 3, CV_32FC1);
	Mat distCoeffs;
	//vector<Mat> rvecs;
	//vector<Mat> tvecs;

	//intrinsic.ptr<float>(0)[0] = 1;
	//intrinsic.ptr<float>(1)[1] = 1;

	//calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs);

	// save cam parameters
	//ofstream intrinsicFile,distCoeffsFile;
	//intrinsicFile.open("intrinsic.txt");
	//intrinsicFile << intrinsic;
	//intrinsicFile.close();
	//distCoeffsFile.open("distCoeffs.txt");
	//distCoeffsFile << distCoeffs;
	//distCoeffsFile.close();

	//SaveMatBinary("intrinsicB", intrinsic);
	//SaveMatBinary("distCoeffsB", distCoeffs);

	LoadMatBinary("intrinsicB", intrinsic);
	LoadMatBinary("distCoeffsB", distCoeffs);


	Mat imageUndistorted;
	Mat mAlignedCam;
	while (1)
	{
		//capture >> image;

		unique_lock<mutex> lock(ir->mtx);
		// item buffer is empty, just wait here.
		//cout << "get frame, write at " << ir->write_position << ", read at  " << ir->read_position << endl;
		while (ir->write_position == ir->read_position)
		{
			//std::cout << "FrameConsumer is waiting for frames...\n";
			(ir->repo_not_empty).wait(lock); // 消费者等待"frame库缓冲区不为空"这一条件发生.
		}
		cv::Mat img = (ir->item_buffer)[ir->read_position].clone();
		cv::Mat IRimg = (ir->IR_buffer)[ir->read_position].clone();
		//cout << "read at " << ir->read_position << endl;
		(ir->read_position)++; // 读取位置后移
		if (ir->read_position >= LEN) // 读取位置若移到最后，则重新置位.
			ir->read_position = 0;

		(ir->repo_not_full).notify_all(); // 通知消费者frame库不为满.
		lock.unlock(); // 解锁.

		image = img.clone();
		undistort(image, imageUndistorted, intrinsic, distCoeffs);

		hconcat(image, imageUndistorted, mAlignedCam);
		imshow("cam calib", mAlignedCam);

		waitKey(1);
	}

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
