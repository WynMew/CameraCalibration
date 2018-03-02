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
	int RVideoCap = 1;
	int LVideoCap = 0;
	int iFrameRow;
	int iFrameCol;
	iFrameRow = 480;
	iFrameCol = 640;


	VideoCapture RCam;
	VideoCapture LCam;

	RCam = VideoCapture(RVideoCap);
	LCam = VideoCapture(LVideoCap);
	RCam.set(CV_CAP_PROP_FRAME_WIDTH, iFrameCol);
	RCam.set(CV_CAP_PROP_FRAME_HEIGHT, iFrameRow);
	RCam.set(CV_CAP_PROP_AUTO_EXPOSURE, 1);
	LCam.set(CV_CAP_PROP_FRAME_WIDTH, iFrameCol);
	LCam.set(CV_CAP_PROP_FRAME_HEIGHT, iFrameRow);
	LCam.set(CV_CAP_PROP_AUTO_EXPOSURE, 1);


	cv::Mat RImg;
	cv::Mat LImg;
	cv::Mat mFrameNow;
	int counter = 0;
	while (1)
	{
		RCam >> RImg;
		LCam >> LImg;

		if (!RImg.empty() && !LImg.empty())
		{
			//transpose(RImg, RImg);
			//transpose(LImg, LImg);
			hconcat(RImg, LImg, mFrameNow);
			imshow("Frame Now", mFrameNow);
			cvWaitKey(1);

			std::ostringstream RFileNameSS;
			RFileNameSS <<"R_"<< counter << ".jpg";
			string RFileName = RFileNameSS.str();

			std::ostringstream LFileNameSS;
			LFileNameSS<<"L_" << counter << ".jpg";
			string LFileName = LFileNameSS.str();
			
			imwrite(RFileName, RImg);
			imwrite(LFileName, LImg);

			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			counter++;
		}

	}


}
