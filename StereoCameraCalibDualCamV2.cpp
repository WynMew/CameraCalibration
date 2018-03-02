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
	vector<vector<Point2f>> RGBpoints;
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
	RGBpoints.push_back(corners);

	found = findChessboardCorners(CalibIRImg, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
	cornerSubPix(grayCalibIRImg, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
	IRpoints.push_back(corners);


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

	double rms = stereoCalibrate(objectpoints, RGBpoints, IRpoints,
			intrinsic, distCoeffs,
			intrinsicIR, distCoeffsIR,
			image.size(), R, T, E, F,
			TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 50, 1e-6),
			CV_CALIB_FIX_INTRINSIC +
			CV_CALIB_USE_INTRINSIC_GUESS
			);

	std::cout << rms << std::endl;

	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];

	stereoRectify(intrinsic, distCoeffs,
		intrinsicIR, distCoeffsIR,
		image.size(), R, T, R1, R2, P1, P2, Q,
		CALIB_ZERO_DISPARITY, 1, image.size(), &validRoi[0], &validRoi[1]);

	Mat rmap[2][2];
	vector<Point2f> allimgpt[2];

	//std::copy(RGBpoints.begin(), RGBpoints.end(), back_inserter(allimgpt[0]));
	//std::copy(IRpoints.begin(), IRpoints.end(), back_inserter(allimgpt[1]));

	Mat gCalibRGBImg, gCalibIRImg;
	cvtColor(CalibRGBImg, gCalibRGBImg, CV_BGR2GRAY);
	cvtColor(CalibIRImg, gCalibIRImg, CV_BGR2GRAY);

	F = findFundamentalMat(Mat(RGBpoints[0]), Mat(IRpoints[0]), FM_8POINT, 0, 0);
	Mat H1, H2;
	stereoRectifyUncalibrated(Mat(RGBpoints[0]), Mat(IRpoints[0]), F, image.size(), H1, H2, 3);


	cv::Mat newRGBImg, newIRImg;

	while (1){
		unique_lock<mutex> lock(ir->mtx);
		// item buffer is empty, just wait here.
		//cout << "get frame, write at " << ir->write_position << ", read at  " << ir->read_position << endl;
		while (ir->write_position == ir->read_position)
		{
			//std::cout << "FrameConsumer is waiting for frames...\n";
			(ir->repo_not_empty).wait(lock); // 消费者等待"frame库缓冲区不为空"这一条件发生.
		}
		newRGBImg = (ir->item_buffer)[ir->read_position].clone();
		newIRImg = (ir->IR_buffer)[ir->read_position].clone();
		//cout << "read at " << ir->read_position << endl;
		(ir->read_position)++; // 读取位置后移
		if (ir->read_position >= LEN) // 读取位置若移到最后，则重新置位.
			ir->read_position = 0;

		(ir->repo_not_full).notify_all(); // 通知消费者frame库不为满.
		lock.unlock(); // 解锁.

		Mat gnewRGBImg, gnewIRImg;
		cvtColor(newRGBImg, gnewRGBImg, CV_BGR2GRAY);
		cvtColor(newIRImg, gnewIRImg, CV_BGR2GRAY);


		cv::Mat rectified1(gnewRGBImg.size(), gnewRGBImg.type());
		cv::warpPerspective(gnewRGBImg, rectified1, H1, gnewRGBImg.size());

		cv::Mat rectified2(gnewIRImg.size(), gnewIRImg.type());
		cv::warpPerspective(gnewIRImg, rectified2, H2, gnewIRImg.size());

		Mat disp, disp8, rec;


		StereoBM sbm;
		sbm.state->SADWindowSize = 25;
		sbm.state->numberOfDisparities = 128;
		sbm.state->preFilterSize = 5;
		sbm.state->preFilterCap = 61;
		sbm.state->minDisparity = 0;
		sbm.state->textureThreshold = 507;
		sbm.state->uniquenessRatio = 0;
		sbm.state->speckleWindowSize = 0;
		sbm.state->speckleRange = 8;
		sbm.state->disp12MaxDiff = 1;
		sbm(rectified1, rectified2, disp);
		normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);

		hconcat(rectified1, rectified2, rec);
		imshow("rectified", rec);
		imshow("disp8", disp8);
		waitKey(1);
	}
	
	//R1 = intrinsic.inv()*H1*intrinsic;
	//R2 = intrinsicIR.inv()*H2*intrinsicIR;
	//P1 = intrinsic;
	//P2 = intrinsicIR;

	//initUndistortRectifyMap(intrinsic, distCoeffs, R1, P1, image.size(), CV_16SC2, rmap[0][0], rmap[0][1]);
	//initUndistortRectifyMap(intrinsicIR, distCoeffsIR, R1, P1, image.size(), CV_16SC2, rmap[1][0], rmap[1][1]);

	//Mat canvas;
	//double sf;
	//int w, h;
	//sf = 600. / max(image.cols, image.rows);
	//w = cvRound(image.cols*sf);
	//h = cvRound(image.rows*sf);
	//canvas.create(h, w * 2, CV_8UC3);

	//Mat rimg;

	//remap(CalibRGBImg, rimg, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
	//Mat canvasPart = canvas(Rect(w*0, 0, w, h));
	//resize(rimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
	//Rect vroi(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
	//	cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));
	//rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);

	//remap(CalibIRImg, rimg, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);
	//canvasPart = canvas(Rect(w * 0, 0, w, h));
	//resize(rimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
	//Rect vroiIR(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
	//	cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));
	//rectangle(canvasPart, vroiIR, Scalar(0, 0, 255), 3, 8);

	//for (int j = 0; j < canvas.rows; j += 16)
	//	line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);

	//imshow("canvas", canvas);
	//waitKey(1);
	//char c = (char)waitKey();
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
