//根据以上代码将其在opencv3.0.0中编写
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"
#include "iostream"
#include"detect.h"


using namespace cv;
using namespace std;


Armor armor;
ArmorDetector armordetector;
//ArmorDetector ArmorDetector;
int main()
{
	//using cvex::distance;
	float light_max_angle_diff_;
	float light_max_height_diff_ratio_; // hdiff / max(r.length, l.length) 
	float light_max_y_diff_ratio_;  // ydiff / max(r.length, l.length) 
	float light_min_x_diff_ratio_;
	float armor_big_armor_ratio;
	float armor_small_armor_ratio;
	float armor_min_aspect_ratio_;
	float armor_max_aspect_ratio_;
	light_max_angle_diff_ = 4.0; //20 7.0灯条最大角差
	light_max_height_diff_ratio_ = 0.4; //0.5 灯条最大高度差比率
	light_max_y_diff_ratio_ = 0.5; //2.0,100 左右灯条中心点y的最大差比率
	light_min_x_diff_ratio_ = 0.1; //0.5,100 左右灯条中心点x的最小差比率
	armor_min_aspect_ratio_ = 1.0;//1.0
	armor_max_aspect_ratio_ = 5.6;//5.0
	Mat frame, mask, dst;
	VideoCapture cap;
	vector<Mat> channels;
	//if (!cap.open("考核视频.avi"))
	if (!cap.open("4.MOV"))
	{
		cout << "video is err" << endl;
		return -1;
	}

	//开操作 掩膜
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	RotatedRect rect;
	while (true)
	{
		cap >> frame;
		if (frame.empty())
		{
			break;
		}
		namedWindow("inputvideo", 0);
		resizeWindow("inputvideo", 800, 500);
		imshow("inputvideo", frame);
		dst = frame.clone();
		cvtColor(frame, frame, COLOR_BGR2HSV);
		inRange(frame, Scalar(100, 130, 180), Scalar(124, 255, 255), mask);
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		//膨胀
		dilate(mask, mask, element);

		//开操作 消除噪点
		morphologyEx(mask, mask, MORPH_OPEN, kernel);
		namedWindow("1", 0);
		resizeWindow("1", 800, 500);
		imshow("1", mask);
		vector<RotatedRect> lightInfos;
		vector<vector<Point>> contours;
		vector<RotatedRect> _armors;
		Rect rect;
		Rect rect1;

		findContours(mask, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
		for (size_t i = 0; i < contours.size(); i++)
		{
			rect = boundingRect(contours[i]);
			if (rect.width > 9 && rect.height > 30 && rect.width < 80 && rect.height < 60)
			{
				RotatedRect rotatedRect = fitEllipse(contours[i]);
				lightInfos.push_back(rotatedRect);
			}
		}

		sort(lightInfos.begin(), lightInfos.end(), [](RotatedRect& a1, RotatedRect& a2) -> bool
		{
			return a1.center.x < a2.center.x;
		});

		for (size_t i = 0; i < lightInfos.size(); i++)
		{//遍历所有灯条进行匹配
			for (size_t j = i + 1; j < lightInfos.size(); j++)
			{
				RotatedRect &leftLight = lightInfos[i];

				RotatedRect &rightLight = lightInfos[j];

				//const LightDescriptor& leftLight = lightInfos[i];
				//const LightDescriptor& rightLight = lightInfos[j];
				/*
				*	Works for 2-3 meters situation
				*	morphologically similar: // parallel
								 // similar height
				*/
				//角差
				float angleDiff_ = abs(leftLight.angle - rightLight.angle);
				//长度差比率
				float LenDiff_ratio = abs(leftLight.size.height - rightLight.size.height) / max(leftLight.size.height, rightLight.size.height);
				//筛选
				if (angleDiff_ > light_max_angle_diff_ ||
					LenDiff_ratio > light_max_height_diff_ratio_)
				{
					continue;
				}


				//using cvex::distance;
				/*
				*	proper location:  y value of light bar close enough
				*			  ratio of length and width is proper
				*/
				//左右灯条相距距离
				//float dis = (leftLight.center.x - rightLight.center.x);
				//float dis = cv::distance(leftLight.center, rightLight.center);
				//左右灯条长度的平均值
				float meanLen = (leftLight.size.height + rightLight.size.height) / 2;
				//左右灯条中心点y的差值
				float yDiff = fabs(leftLight.center.y - rightLight.center.y);
				//y差比率
				float yDiff_ratio = yDiff / meanLen;
				//左右灯条中心点x的差值
				float xDiff = fabs(leftLight.center.x - rightLight.center.x);
				//x差比率
				float xDiff_ratio = xDiff / meanLen;
				//相距距离与灯条长度比值
				float ratio = xDiff / meanLen;
				//筛选
				if (yDiff_ratio > light_max_y_diff_ratio_ ||
					xDiff_ratio < light_min_x_diff_ratio_ ||
					ratio > armor_max_aspect_ratio_ ||
					ratio < armor_min_aspect_ratio_)
				{
					continue;
				}

				//Rect armorRect = boundingRect(leftLight.points);
				RotatedRect armorRotatedRect = armordetector.boundingRotatedRect(leftLight, rightLight);
				rect1 = armorRotatedRect.boundingRect();
				rectangle(dst, rect1, Scalar(0, 0, 255), 5);
			}

		}
		namedWindow("dst", 0);
		resizeWindow("dst", 800, 500);
		imshow("dst", dst);


		if (waitKey(2) > 0)
			break;
	}
	cap.release();
	destroyAllWindows();
	return 0;
}
