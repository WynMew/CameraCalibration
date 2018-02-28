#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "VideoFaceDetector.h"
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>


using namespace cv;
using namespace std;

void main(int argc, char** argv)
{
	int RGBVideoCap = 1;
	int IRVideoCap = 0;

	VideoFaceDetector *VFD = new VideoFaceDetector(RGBVideoCap, IRVideoCap);
	std::thread VideoCaptureTask = VFD->VideoCaptureThread();
	std::thread FaceProposeTask = VFD->FaceProposerThread();
	std::thread LocalVerifyTask = VFD->LocalVerifierThread();

	VideoCaptureTask.join();
	FaceProposeTask.join();
	LocalVerifyTask.join();
	delete VFD;
}


#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "VideoFaceDetector.h"
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>


using namespace cv;
using namespace std;


void main(int argc, char** argv)
{
	int RGBVideoCap = 1;
	int IRVideoCap = 0;
	int iFrameRow;
	int iFrameCol;
	iFrameRow = 480;
	iFrameCol = 640;


	VideoCapture cap;
	VideoCapture IR;

	cap = VideoCapture(RGBVideoCap);
	IR = VideoCapture(IRVideoCap);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, iFrameCol);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, iFrameRow);
	cap.set(CV_CAP_PROP_AUTO_EXPOSURE, 1);
	IR.set(CV_CAP_PROP_FRAME_WIDTH, iFrameCol);
	IR.set(CV_CAP_PROP_FRAME_HEIGHT, iFrameRow);
	IR.set(CV_CAP_PROP_AUTO_EXPOSURE, 1);


	cv::Mat RGBimg;
	cv::Mat IRimg;
	cv::Mat mFrameNow;
	int counter = 0;
	while (1)
	{
		cap >> RGBimg;
		IR >> IRimg;

		if (!RGBimg.empty() && !IRimg.empty())
		{
			transpose(RGBimg, RGBimg);
			transpose(IRimg, IRimg);
			hconcat(RGBimg, IRimg, mFrameNow);
			imshow("Frame Now", mFrameNow);
			cvWaitKey(1);

			std::ostringstream RGBfileNameSS;
			RGBfileNameSS <<"RGB_"<< counter << ".jpg";
			string RGBFileName = RGBfileNameSS.str();

			std::ostringstream IRfileNameSS;
			IRfileNameSS<<"IR_" << counter << ".jpg";
			string IRFileName = IRfileNameSS.str();
			
			imwrite(RGBFileName, RGBimg);
			imwrite(IRFileName, IRimg);

			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			counter++;
		}

	}


}
