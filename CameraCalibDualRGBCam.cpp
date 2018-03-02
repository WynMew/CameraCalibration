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
	cv::Mat RItem_buffer[LEN];
	cv::Mat LItem_buffer[LEN];
	size_t read_position;
	size_t write_position;
	std::mutex mtx;
	std::condition_variable repo_not_full;
	std::condition_variable repo_not_empty;
} gFrameRepository;


typedef struct FrameRepository FrameRepository;

VideoCapture RCap;
VideoCapture LCap;


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

	(ir->RItem_buffer)[ir->write_position] = item.clone();
	(ir->LItem_buffer)[ir->write_position] = IRimg.clone();
	//cout << "write at " << ir->write_position << endl;
	(ir->write_position)++; // 写入位置后移.

	if (ir->write_position == LEN) // 写入位置若是在队列最后则重新设置为初始位置.
		ir->write_position = 0;

	(ir->repo_not_empty).notify_all(); // 通知消费者frame库不为空.
	lock.unlock();
}

void CameraCapture()
{
	cv::Mat RImg;
	cv::Mat LImg;
	while (1)
	{
		RCap >> RImg;
		LCap >> LImg;
		if (!RImg.empty() && !LImg.empty()){
			//cv::cvtColor(img,img, CV_BGR2RGB);
			ProduceFrameItem(&gFrameRepository, RImg, LImg);// write into frame repo
		}
	}
}

void camcalib()
{
	FrameRepository *ir = &gFrameRepository;
	cv::Mat RImage;
	cv::Mat LImage;
	cv::Mat intrinsicR = Mat(3, 3, CV_32FC1);
	cv::Mat distCoeffsR;
	cv::Mat intrinsicL = Mat(3, 3, CV_32FC1);
	cv::Mat distCoeffsL;

	LoadMatBinary("intrinsicR", intrinsicR);
	LoadMatBinary("distCoeffsR", distCoeffsR);
	LoadMatBinary("intrinsicL", intrinsicL);
	LoadMatBinary("distCoeffsL", distCoeffsL);


	Mat RImgUndistorted;
	Mat mRAlignedCam;
	Mat LImgUndistorted;
	Mat mLAlignedCam;

	while (1)
	{
		unique_lock<mutex> lock(ir->mtx);
		// item buffer is empty, just wait here.
		//cout << "get frame, write at " << ir->write_position << ", read at  " << ir->read_position << endl;
		while (ir->write_position == ir->read_position)
		{
			//std::cout << "FrameConsumer is waiting for frames...\n";
			(ir->repo_not_empty).wait(lock); // 消费者等待"frame库缓冲区不为空"这一条件发生.
		}
		cv::Mat RImg = (ir->RItem_buffer)[ir->read_position].clone();
		cv::Mat LImg = (ir->LItem_buffer)[ir->read_position].clone();
		//cout << "read at " << ir->read_position << endl;
		(ir->read_position)++; // 读取位置后移
		if (ir->read_position >= LEN) // 读取位置若移到最后，则重新置位.
			ir->read_position = 0;

		(ir->repo_not_full).notify_all(); // 通知消费者frame库不为满.
		lock.unlock(); // 解锁.

		RImage = RImg.clone();
		LImage = LImg.clone();
		undistort(RImage, RImgUndistorted, intrinsicR, distCoeffsR);
		undistort(LImage, LImgUndistorted, intrinsicL, distCoeffsL);
		hconcat(RImage, RImgUndistorted, mRAlignedCam);
		imshow("RCam calib", mRAlignedCam);
		hconcat(LImage, LImgUndistorted, mLAlignedCam);
		imshow("LCam calib", mLAlignedCam);

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
		ir->RItem_buffer[i] = matrix.clone(); // write in zeros
		ir->LItem_buffer[i] = matrix.clone();
	}
}


int main()
{
	RCap = VideoCapture(0);
	RCap.set(CV_CAP_PROP_FRAME_WIDTH, iFrameCol);
	RCap.set(CV_CAP_PROP_FRAME_HEIGHT, iFrameRow);
	RCap.set(CV_CAP_PROP_AUTO_EXPOSURE, 1);

	LCap = VideoCapture(1);
	LCap.set(CV_CAP_PROP_FRAME_WIDTH, iFrameCol);
	LCap.set(CV_CAP_PROP_FRAME_HEIGHT, iFrameRow);
	//cap.set(CV_CAP_PROP_EXPOSURE, -3); 
	LCap.set(CV_CAP_PROP_AUTO_EXPOSURE, 1);

	InitFrameRepository(&gFrameRepository);
	std::thread CameraCaptureTask(CameraCapture);
	std::thread CameraCalibTask(camcalib);

	CameraCaptureTask.join();
	CameraCalibTask.join();

	return 0;
}
