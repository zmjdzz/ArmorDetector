#include"detect.h"
using namespace cv;
//
// Create by FYQ on 2019/8/11
//


/*--------------------------------------------------------------------
\@brief     ArmorLight constructor
\@param     light bounding rectangle
\@param     light angle
*------------------------------------------------------------------*/
ArmorLight::ArmorLight(const cv::RotatedRect& rotatedRect, cv::Point2f *points) :cv::RotatedRect(rotatedRect.center,
	rotatedRect.size,
	rotatedRect.angle)
{
	rotateRectPoints = { points[0], points[1], points[2], points[3] };
}

/*--------------------------------------------------------------------
\@brief         ArmorLight destructor
*------------------------------------------------------------------*/
ArmorLight::~ArmorLight()
{
}

/*--------------------------------------------------------------------
\@brief         Armor constructor
\@param         armor image
\@param         armor center
\@param         armor bounding rectangle
\@param         armor angle
*------------------------------------------------------------------*/
Armor::Armor(cv::Mat& armorImg, cv::RotatedRect& rotatedRect) :armorImg(armorImg)
{
	this->armorRotatedRect.size = rotatedRect.size;
	this->armorRotatedRect.center = rotatedRect.center;
	this->armorRotatedRect.angle = rotatedRect.angle;
}

/*--------------------------------------------------------------------
\@brief         copy constructor
\@param         copy object
*------------------------------------------------------------------*/
Armor::Armor(Armor& armor)
{
	*this = armor;
}

/*--------------------------------------------------------------------
\@brief         move constructor
\@param         move object
*------------------------------------------------------------------*/
Armor::Armor(Armor&& armor) noexcept
{
	*this = std::move(armor);
}

/*--------------------------------------------------------------------
\@brief         copy assignment
\@param         move object
*------------------------------------------------------------------*/
Armor& Armor::operator=(Armor& armor)
{
	this->armorRotatedRect = armor.armorRotatedRect;
	this->armorImg = armor.armorImg;
	return *this;
}

/*--------------------------------------------------------------------
\@brief         move assignment
\@param         move object
*------------------------------------------------------------------*/
Armor& Armor::operator=(Armor&& armor) noexcept
{
	this->armorRotatedRect = std::move(armor.armorRotatedRect);
	this->armorImg = std::move(armor.armorImg);
	return *this;
}

/*--------------------------------------------------------------------
\@brief     destructor
*------------------------------------------------------------------*/
Armor::~Armor()
{
}

/*--------------------------------------------------------------------
\@brief         setting enemy  light color
\@param         enemy light color
*------------------------------------------------------------------*/
void ArmorDetector::setEnemyColor(Color enemyColor)
{
	_enemyColor = enemyColor;
}

/*--------------------------------------------------------------------
\@brief         setting the area of roi
\@param         the input image
*------------------------------------------------------------------*/
void ArmorDetector::setRoi(cv::Mat &inputImg)
{
	if (_lastRotateRect.center.x <= 0 ||
		_lastRotateRect.center.y <= 0 /*||
		_lastRotateRect.center.x >= inputImg.size().width ||
		_lastRotateRect.center.y >= inputImg.size().height*/)
	{
		_detecArea.x = 0;
		_detecArea.y = 0;
		return;
	}

	cv::Point finalAimCenter = _lastRotateRect.center;
	cv::Rect rect = _lastRotateRect.boundingRect();

	int detecRectWidth = rect.width * 2;
	int detecRectHeight = rect.height * 2;

	int tlx = std::max(finalAimCenter.x - detecRectWidth, 0);
	int tly = std::max(finalAimCenter.y - detecRectHeight, 0);
	cv::Point tl(tlx, tly);

	int brx = std::min(finalAimCenter.x + detecRectWidth, inputImg.cols);
	int bry = std::min(finalAimCenter.y + detecRectHeight, inputImg.rows);
	cv::Point br(brx, bry);

	_detecArea = cv::Rect(tl, br);
	makeRectSafe(_detecArea, inputImg.size());

	inputImg = inputImg(_detecArea);

}

/*---------------------------------------------------------------------------------
\@brief         distill the enemy color region
\@param         the input image
\@param[OUT]    the output image which is 8-bit binary image
\@param         chose detection mode,true is hsv ,false is color subtract
*---------------------------------------------------------------------------------*/
void ArmorDetector::extractColor(cv::Mat& inputImg, cv::Mat& outputImg, bool detectionMode)
{
	static const cv::Scalar hsv_red_floor(0, 43, 220);//下限
	static const cv::Scalar hsv_red_ceil(40, 255, 255);//上限
	static const cv::Scalar hsv_blue_floor(80, 30, 220);
	static const cv::Scalar hsv_blue_ceil(100, 255, 255);

	if (detectionMode)
	{
		cv::Mat hsv;
		cv::cvtColor(inputImg, hsv, cv::COLOR_BGR2HSV);
		switch (_enemyColor)
		{
		case (Color::RED): ////red hsv
		{
			cv::inRange(hsv, hsv_red_floor, hsv_red_ceil, outputImg);
			break;
		}
		case (Color::BLUE): ////blue hsv
		{
			cv::inRange(hsv, hsv_blue_floor, hsv_blue_ceil, outputImg);
			break;
		}
		}
	}
	else
	{
		static std::vector<cv::Mat> bgr;
		cv::split(inputImg, bgr);
		switch (_enemyColor)
		{

		case (Color::RED): ////red color
		{
			outputImg = bgr[Color::RED] - bgr[Color::BLUE];
			break;
		}
		case (Color::BLUE): ////blue color
		{
			outputImg = bgr[Color::BLUE] - bgr[Color::RED];
			break;
		}
		}
		cv::threshold(outputImg, outputImg, _detectParam.extractColorThreshold, 255, cv::THRESH_BINARY);
		bgr.clear();
	}
}

/*--------------------------------------------------------------------
\@brief         distill the bightness region
\@param         the input image
\@param[OUT]    the output image which is 8-bit binary image
*------------------------------------------------------------------*/
void ArmorDetector::extractBrightness(cv::Mat& inputImg, cv::Mat& outputImg)
{
	cv::cvtColor(inputImg, outputImg, cv::COLOR_BGR2GRAY);
	cv::threshold(outputImg, outputImg, _detectParam.extractBrightnessThreshold, 255, cv::THRESH_BINARY);
}

/*--------------------------------------------------------------------
\@brief         find the correct armor light
\@param         the extracted color binary image
\@param         the extracted birght binary image
\@param[OUT]    the set of armor light which was found
*------------------------------------------------------------------*/
bool ArmorDetector::findArmoredLights(const cv::Mat& binColorImg,
	const cv::Mat& binBrightImg
)
{
	assert(binColorImg.channels() == 1 && binBrightImg.channels() == 1);

	static const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	static const cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 1));
	static std::vector<std::vector<cv::Point>> colorContours;
	static std::vector<std::vector<cv::Point>> brightContours;
	static cv::Point2f rotateRectPoints[4];
	cv::morphologyEx(binColorImg, binColorImg, cv::MORPH_OPEN, kernel);
	// cv::morphologyEx(binBrightImg, binBrightImg, cv::MORPH_OPEN,kernel1);
	 //使用回上颜色包围亮度的方法，如果使用&去合并两幅图像会导致远距离的灯条过小识别不够稳定
	cv::findContours(binColorImg, colorContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	cv::findContours(binBrightImg, brightContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	//    cv::Mat test;
	//    cv::cvtColor(binColorImg,test, cv::COLOR_GRAY2BGR);
	//    for(size_t i = 0; i < brightContours.size(); i++)
	//    {
	//
	//            cv::circle(test, brightContours[i][brightContours[i].size()/4], 1, cv::Scalar(0,0,255));
	//
	//    }
	//    cv::drawContours(test, colorContours, -1, cv::Scalar(0,255,0));
	//    cv::drawContours(test, brightContours, -1, cv::Scalar(255,0,0));
	for (size_t i = 0; i < brightContours.size(); i++)
	{
		for (size_t j = 0; j < colorContours.size(); j++)
		{
			int midContours = brightContours[i].size() / 2;
			if (cv::pointPolygonTest(colorContours[j], brightContours[i][midContours], false) > -1 /*||
			   cv::pointPolygonTest(brightContours[i], colorContours[j][0],false) > -1*/)
			{

				if (colorContours[j].size() < 5)
				{
					//Log_Info << "begin";
					continue;
				}
				cv::RotatedRect rotatedRect = cv::fitEllipse(colorContours[j]);
				//the rotateRect angle value range in [0, 180], no negative value
				if (rotatedRect.angle >= _detectParam.lightAngleThreshold && rotatedRect.angle <= 180 - _detectParam.lightAngleThreshold)
				{
					//Log_Info << "1";
					continue;
				}

				if (rotatedRect.size.height / rotatedRect.size.width < _detectParam.minLightAspectRatio)
				{
					//Log_Info << "2";
					continue;
				}

				ajustAngle(rotatedRect);
				rotatedRect.points(rotateRectPoints);
				_armorLights.emplace_back(rotatedRect, rotateRectPoints);
			}
		}
	}
	//IMSHOW("gray", test);
	imshow("binry", binColorImg);
	imshow("brigness", binBrightImg);
	colorContours.clear();
	brightContours.clear();
	waitKey(1);
	return _armorLights.size() >= 2;
}

/*--------------------------------------------------------------------
\@brief         find the correct armor
\@param         the source image
*------------------------------------------------------------------*/
bool ArmorDetector::findArmors(cv::Mat &srcImg)
{
	////five inspection items: width, height, height ratio, angleDiff, aspect ration
	int armorLightsSize = static_cast<int>(_armorLights.size());
	////sort by armor light x
	std::sort(_armorLights.begin(), _armorLights.end(), [](ArmorLight& a1, ArmorLight& a2) -> bool
	{
		return a1.center.x < a2.center.x;
	});
	for (int i = 0; i < armorLightsSize - 1; i++)
	{
		for (int j = i + 1; j < armorLightsSize; j++)
		{
			ArmorLight& left = _armorLights[i];
			ArmorLight& right = _armorLights[j];

			//左右灯条角度不能相差过大
			float angleDiff = std::fabs(std::fabs(left.angle) - std::fabs(right.angle));

			if (angleDiff >= _detectParam.lightAngleDiff)
			{
				//Log_Info << "3";
				continue;
			}

			//check lights slope
			float x = right.center.x - left.center.x;
			float y = right.center.y - left.center.y;

			if (x < (left.size.width + right.size.width) * 2)
			{
				//Log_Info << "4";
				continue;
			}

			float hightMax = std::max(left.size.height, right.size.height);
			if (std::fabs(left.center.y - right.center.y) > hightMax * 0.8)
			{
				//Log_Info << "5";
				continue;
			}

			//check length and width ratio
			//ratio should be in the armor width height ratio
			cv::RotatedRect armorRotatedRect = boundingRotatedRect(left, right);
			//  Log_Info << "left height : " << (left.size.height > left.size.width? left.size.height : left.size.width);
			ajustAngle(armorRotatedRect);
			float width = std::max(armorRotatedRect.size.width, armorRotatedRect.size.height);
			float height = std::min(armorRotatedRect.size.width, armorRotatedRect.size.height);
			float ratio = width / height;
			//aspect ratio
			if (ratio < _detectParam.minAspectRatio || ratio > _detectParam.maxAspectRatio)
			{
				//Log_Info << "6";
				break;
			}
			cv::Rect roi = armorRotatedRect.boundingRect();
			makeRectSafe(roi, srcImg.size());
			cv::Mat armorImg(srcImg, roi);
			cv::cvtColor(armorImg, armorImg, cv::COLOR_BGR2GRAY);
			_armors.emplace_back(armorImg, armorRotatedRect);
		}
	}
	return !_armors.empty();
}

/*--------------------------------------------------------------------
\@brief         ajust the rotated rectangle to the range of range (-90, 90)
\@param         the rotated rectangle which need to be ajust
*------------------------------------------------------------------*/
void ArmorDetector::ajustAngle(cv::RotatedRect& rect)
{
	if (rect.angle >= 90)
	{
		rect.angle -= 180;
	}
}

/*--------------------------------------------------------------------
\@brief         the armor image was transformed into perspective
\@param         the input image
\@param[OUT]    the output image
\@param         the rectangle of armor
\@param         the left armor light
\@param         the right armor light
*------------------------------------------------------------------*/
void ArmorDetector::armorPerspectiveTrans(cv::Mat &srcImg, cv::Mat& armorImg, cv::Rect2f& armorRoi, ArmorLight& leftLight, ArmorLight& rightLight)
{
	static cv::Mat transferMatrix;
	static std::vector<cv::Point2f> srcPoint(4);
	static std::vector<cv::Point2f> dstPoint = { cv::Point2f(0, 0),      //top left
												cv::Point2f(0,50),      //bottom left
												cv::Point2f(50, 0),     //top right
												cv::Point2f(50, 50) };   //bottom right

	srcPoint[0] = leftLight.rotateRectPoints[1];    //left light top left
	srcPoint[1] = leftLight.rotateRectPoints[0];    //left light bottom left
	srcPoint[2] = rightLight.rotateRectPoints[2];   //right light top right
	srcPoint[3] = rightLight.rotateRectPoints[3];   //right light bottom right

	transferMatrix = cv::getPerspectiveTransform(srcPoint, dstPoint);
	cv::warpPerspective(srcImg, armorImg, transferMatrix, cv::Size(armorRoi.width, armorRoi.height));
}

/*--------------------------------------------------------------------
\@brief         make sure the rectangle in the image
\@param         the  rectangle that need to be jugded
\@param         the image size
\@return        safe of not
*------------------------------------------------------------------*/
bool ArmorDetector::makeRectSafe(cv::Rect &rect, cv::Size size)
{
	if (rect.width <= 0 || rect.height <= 0)
	{
		// 如果发现矩形是空的，则返回false
		return false;
	}

	if (rect.x < 0)
	{
		rect.x = 0;
	}

	if (rect.x + rect.width > size.width)
	{
		rect.width = size.width - rect.x;
	}

	if (rect.y < 0)
	{
		rect.y = 0;
	}
	if (rect.y + rect.height > size.height)
	{
		rect.height = size.height - rect.y;
	}

	return true;
}

/*--------------------------------------------------------------------
\@brief         use the lights to fitting the bouding rotated rectangle
\@param         left light
\@param         right light
\return         the fitting rotated rectangle
*------------------------------------------------------------------*/
cv::RotatedRect ArmorDetector::boundingRotatedRect(const cv::RotatedRect &left, const cv::RotatedRect &right)
{

	const cv::Point & pl = left.center, &pr = right.center;
	cv::Point2f center = (pl + pr) / 2.0;
	double width_l = left.size.width;
	double width_r = right.size.width;
	double height_l = left.size.height;
	double height_r = right.size.height;
	float width = std::sqrt((pl.x - pr.x)*(pl.x - pr.x) + (pl.y - pr.y)*(pl.y - pr.y)) - (width_l + width_r) / 2.0;
	float height = std::max(height_l, height_r);
	float angle = std::atan2(right.center.y - left.center.y, right.center.x - left.center.x);
	return cv::RotatedRect(center, cv::Size2f(width, height), angle * 180 / CV_PI);
}

/*--------------------------------------------------------------------
\@brief         make the decision of striking
\@return        the armor that decide to attack
*------------------------------------------------------------------*/
Armor ArmorDetector::strikingDecision()
{
	if (_armors.size() == 1)
	{
		return _armors[0];
	}
	std::sort(_armors.begin(), _armors.end(),
		[](Armor &a1, Armor &a2)-> bool
	{
		//float a1Height = std::min(a1.armorRotatedRect.size.height, a1.armorRotatedRect.size.width);
		//float a2Height = std::min(a2.armorRotatedRect.size.height, a2.armorRotatedRect.size.width);
		float a1Area = a1.armorRotatedRect.size.height * a1.armorRotatedRect.size.width;
		float a2Area = a2.armorRotatedRect.size.height * a2.armorRotatedRect.size.width;
		return a1Area > a2Area;
	});
	size_t finalIndex = 0;
	float finalDistance = std::sqrt(((_armors[finalIndex].armorRotatedRect.center.x - _detectParam.imageCenterX) * \
		(_armors[finalIndex].armorRotatedRect.center.x - _detectParam.imageCenterX)) + \
		((_armors[finalIndex].armorRotatedRect.center.y - _detectParam.imageCenterY) * \
		(_armors[finalIndex].armorRotatedRect.center.y - _detectParam.imageCenterY)));

	for (size_t i = 1; i < _armors.size(); ++i)
	{
		float distance = std::sqrt(((_armors[i].armorRotatedRect.center.x - _detectParam.imageCenterX) * \
			(_armors[i].armorRotatedRect.center.x - _detectParam.imageCenterX)) + \
			((_armors[i].armorRotatedRect.center.y - _detectParam.imageCenterY) * \
			(_armors[i].armorRotatedRect.center.y - _detectParam.imageCenterY)));

		if (std::fabs(_armors[i].armorRotatedRect.angle) < std::fabs(_armors[finalIndex].armorRotatedRect.angle))
		{
			if (std::fabs(_armors[finalIndex].armorRotatedRect.angle) - std::fabs(_armors[i].armorRotatedRect.angle) < 3.0f)
			{
				if (finalDistance > distance)
				{
					finalDistance = distance;
					finalIndex = i;
				}
			}
			else
			{
				finalIndex = i;
			}
		}
	}

	return _armors[finalIndex];
}

/*--------------------------------------------------------------------
\@brief         detect the armor
\@param         the input image
*------------------------------------------------------------------*/
Armor ArmorDetector::detect(cv::Mat &srcImg)
{
	Armor finalArmor;
	cv::Mat binColor;
	cv::Mat binBright;
	cv::Mat inputImage = srcImg;
	setRoi(inputImage);
	extractColor(inputImage, binColor, false);
	extractBrightness(inputImage, binBright);
	bool flag = findArmoredLights(binColor, binBright);
	if (findArmors(inputImage))
	{
		finalArmor = strikingDecision();
#ifdef DEBUG
		cv::Point2f p[4];
		finalArmor.armorRotatedRect.points(p);
		for (int i = 0; i < 4; ++i)
		{

			cv::line(inputImage, p[i], p[(i + 1) % 4], cv::Scalar(0, 255, 0), 1);
			//cv::putText(src, ss.str(), p[2], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 0.1);

		}
#endif
		//Log_Info << "center x :" << finalArmor.armorRotatedRect.center.x;
		//Log_Info << "center y :" << finalArmor.armorRotatedRect.center.y;
		finalArmor.armorRotatedRect.center.x += _detecArea.x;
		finalArmor.armorRotatedRect.center.y += _detecArea.y;
		_lastRotateRect = finalArmor.armorRotatedRect;
		_lostCount = 0;
	}
	else
	{
		//Log_Info << flag;
		++_lostCount;
		if (_lostCount < 8)
		{
			_lastRotateRect.size = cv::Size2f(_lastRotateRect.size.width, _lastRotateRect.size.height);
		}
		if (_lostCount == 10)
		{
			_lastRotateRect.size = cv::Size2f(_lastRotateRect.size.width * 1.5, _lastRotateRect.size.height * 1.5);
		}

		if (_lostCount == 12)
		{
			_lastRotateRect.size = cv::Size2f(_lastRotateRect.size.width * 2, _lastRotateRect.size.height * 2);
		}

		if (_lostCount == 15)
		{
			_lastRotateRect.size = cv::Size2f(_lastRotateRect.size.width * 1.5, _lastRotateRect.size.height * 1.5);
		}

		if (_lostCount == 18)
		{
			_lastRotateRect.size = cv::Size2f(_lastRotateRect.size.width * 1.5, _lastRotateRect.size.height * 1.5);
		}

		if (_lostCount > 33)
		{
			_lastRotateRect = cv::RotatedRect();
		}
	}

	//imshow("in", inputImage);
	//waitKey(1);
	_armorLights.clear();
	_armors.clear();
	return finalArmor;
}
