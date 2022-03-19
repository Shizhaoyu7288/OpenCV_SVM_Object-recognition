
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <memory>
#include <ctime>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;
using namespace ml;

//全局变量
Size sampleSize(160, 160);//样本的大小
int train_samples = 96;
int classes = 2;
Mat trainData;
Mat trainClasses;

//typedef struct {
//	int number;
//	string name[3] = {"ball","cube","cone"};
//	float position[10][2];
//}person_t;

//申明全局函数

//Mat readImageSaveContour(Mat src);
//void getData();

//Mat readImageSaveContour(Mat src)
//{
//	Mat imageWhite;
//	Mat reverseBinaryImage;
//	threshold(src, imageWhite, 100, 255, 8);
//	/*imageWhite = 255 - imageWhite;*/
//	bitwise_not(imageWhite, reverseBinaryImage);
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//	findContours(reverseBinaryImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
//	//最大轮廓
//	double maxarea = 0;
//	int maxAreaIdx = 0;
//	for (int index = contours.size() - 1; index >= 0; index--)
//	{
//		double tmparea = fabs(contourArea(contours[index]));
//		if (tmparea>maxarea)
//		{
//			maxarea = tmparea;
//			maxAreaIdx = index;
//		}
//	}
//	Rect rectangleTem = boundingRect(contours[maxAreaIdx]);
//	Mat image;
//	image = Mat::zeros(src.size(), CV_8UC1);
//	drawContours(image, contours, 0, Scalar(255), 2, 8, hierarchy);
//	//Rect newRectangleTem(rectangleTem.x - 1, rectangleTem.y - 1, rectangleTem.width + 2, rectangleTem.height+2);
//	Mat tem = image(rectangleTem);
//	return tem;
//}
//void getData()
//{
//	trainData.create(train_samples*classes, sampleSize.width*sampleSize.height, CV_32FC1);
//	trainClasses.create(train_samples*classes, 1, CV_32SC1);
//	Mat src_image;
//	char file[255];
//	int i, j;
//	for (i = 0; i<classes; i++)
//	{
//		for (j = 0; j< train_samples; j++)
//		{
//			//	sprintf(file, "E:/OpenCV/predictShape/predictShape/samples/s%d/%d.png", i, j);
//			//训练图像所在文件夹
//			std::string path = "E:/OpenCV/predictShape/predictShape/samples/s";
//			char temp1[256];
//			char temp2[256];
//			sprintf_s(temp1, "%d", i);
//			sprintf_s(temp2, "%d", j);
//			path = path + temp1 + "/" + temp2 + ".png";
//			src_image = imread(path, 0);
//			if (src_image.empty())
//			{
//				printf("Error: Can't load image %s\n", path);
//				//exit(-1);
//			}
//			Mat image = readImageSaveContour(src_image);
//			Mat imageNewSize;
//			resize(image, imageNewSize, sampleSize, CV_INTER_LINEAR);
//			image.release();
//			image = imageNewSize.reshape(1, 1);
//			image.convertTo(trainData(Range(i*train_samples + j, i*train_samples + j + 1), Range(0, trainData.cols)), CV_32FC1);
//			trainClasses.at<float>(i*train_samples + j, 0) = i;
//		}
//	}
//}


int main() {
	int start_svm = clock();
	Ptr<SVM> model = Algorithm::load<SVM>("E:/研电赛/mycode/study/study2/6_9_train/SVM_TRAIN_DATA.xml");
	cout << "开始程序，svm程序执行过程耗时：" << clock() - start_svm << "毫秒" << endl;
	//Mat src = imread("E:/研电赛/mycode/study/study2/6_9_test/test/5.jpg");
	VideoCapture cap(0);//打开摄像头
	if (!cap.isOpened())
	{
		return -1;
	}

	bool stop = false;
	while (!stop) {
		int start_t = clock();
		Mat src;
		cap >> src;
		int start = clock();
		//src = imread("E:/研电赛/mycode/study/study2/6_9_test/test/5.jpg");
		//VideoCapture cap(0);
		//if (!cap.isOpened())
		//{
		//	return -1;
		//}

		//bool stop = false;
		//while (!stop){
		//	Mat src;
		//	cap >> src;
		//namedWindow("result", CV_WINDOW_AUTOSIZE);
		//imshow("result", src);
		Mat show;
		int start_svm1 = clock();
		cvtColor(src, show, CV_BGR2GRAY);//灰度化
		cout << "灰度程序执行过程耗时：" << clock() - start_svm1 << "毫秒" << endl;
		int start_svm2 = clock();
		GaussianBlur(show, show, Size(7, 7), 1.5, 1.5);//高斯滤波
		cout << "高斯滤波程序执行过程耗时：" << clock() - start_svm2 << "毫秒" << endl;
		Mat imageWhite;
		start_svm = clock();
		threshold(show, imageWhite, 0, 255, CV_THRESH_BINARY);
		cout << "二值化程序执行过程耗时：" << clock() - start_svm << "毫秒" << endl;
		bitwise_not(imageWhite, imageWhite);
		/*imageWhite = 255 - imageWhite;*/
		//namedWindow("result", CV_WINDOW_AUTOSIZE);
		//imshow("result", imageWhite);
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		start_svm = clock();
		findContours(imageWhite, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		cout << "寻找边界程序执行过程耗时：" << clock() - start_svm << "毫秒" << endl;
		/*	移除过长或过短的轮廓 */
		start_svm = clock();
		int cmin = 30; //最小轮廓长度  
					   //int cmax = 500;    //最大轮廓  
		double area;
		double minarea = 2000;
		double maxarea = 0;
		vector<vector<Point>>::const_iterator itc = contours.begin();
		while (itc != contours.end())
		{
			/*	if (itc->size() < cmin || itc->size() > cmax)*/
			if (itc->size() < cmin)
				itc = contours.erase(itc);
			else
				++itc;

		}
		vector<vector<Point> >::iterator itr;  //轮廓迭代器
		itr = contours.begin();             //使用迭代器去除噪声轮廓
		while (itr != contours.end())
		{
			area = contourArea(*itr);       //获得轮廓面积
			if (area<minarea)               //删除较小面积的轮廓 
			{
				itr = contours.erase(itr);  //itr一旦erase，需要重新赋值
			}
			else
			{
				itr++;
			}
			if (area>maxarea)              //寻找最大轮廓
			{
				maxarea = area;
			}
		}
		/// 对每个找到的轮廓创建可倾斜的边界框和椭圆
		vector<RotatedRect> minRect(contours.size());
		/* vector<RotatedRect> minEllipse( contours.size() );*/

		for (int i = 0; i < contours.size(); i++)
		{
			minRect[i] = minAreaRect(Mat(contours[i]));
			/*if( contours[i].size() > 5 )
			{ minEllipse[i] = fitEllipse( Mat(contours[i]) ); }*/
		}

		/// 绘出轮廓及其可倾斜的边界框和边界椭圆
		/*Mat drawing = Mat::zeros( result_erase.size(), CV_8UC3 );*/
		for (int i = 0; i< contours.size(); i++)//画出绿色边界框
		{
			Point2f rect_points[4]; minRect[i].points(rect_points);
			for (int j = 0; j < 4; j++)
				line(src, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 255, 0), 2, CV_AA);
		}
		cout << "清除无效边界程序执行过程耗时：" << clock() - start_svm << "毫秒" << endl;
		for (int index = contours.size() - 1; index >= 0; index--)
		{
			start_svm = clock();
			int start_i = clock();
			Rect rectangleTem = boundingRect(contours[index]);
			Mat image;
			image = Mat::zeros(src.size(), CV_8UC1);
			drawContours(image, contours, index, Scalar(255), 2, 8);
			Mat tem = image(rectangleTem);
			Mat imageNewSize;
			resize(tem, imageNewSize, sampleSize, CV_INTER_LINEAR);
			image.release();
			image = imageNewSize.reshape(1, 1);
			image.convertTo(image, CV_32FC1);//图像归一化
			//Ptr<SVM> model = Algorithm::load<SVM>("E:/研电赛/mycode/study/study2/6_9_train/SVM_TRAIN_DATA.xml");
			//cout << "svm程序执行过程耗时：" << clock() - start_svm << "毫秒" << endl;
			/*model->predict(image, trainClasses);
			int response = trainClasses.at<float>(0, 0);*/
			int response = (int)model->predict(image);
			char number[256];
			sprintf_s(number, "Index=%d", contours.size() - index);
			if (response == 0)
			{
				cout << "    ball" << endl;
				string str = " ball," + (string)number;
				putText(src, str, Point(rectangleTem.x + rectangleTem.width / 2, rectangleTem.y + rectangleTem.height / 2),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
				cout << "找到小球程序执行过程耗时：" << clock() - start_svm << "毫秒" << endl;
			}
			else if (response == 1)
			{
				cout << "    cube" << endl;
				string str = " cube," + (string)number;
				putText(src, str, Point(rectangleTem.x + rectangleTem.width / 2, rectangleTem.y + rectangleTem.height / 2),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
				cout << "找到方块程序执行过程耗时：" << clock() - start_svm << "毫秒" << endl;
			}
			else if (response == 2)
			{
				cout << "    cone" << endl;
				string str = " cone," + (string)number;
				putText(src, str, Point(rectangleTem.x + rectangleTem.width / 2, rectangleTem.y + rectangleTem.height / 2),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
				cout << "找到圆锥程序执行过程耗时：" << clock() - start_svm << "毫秒" << endl;
			}
			start_svm = clock();
			cv::Moments Mom;
			Mom = cv::moments(contours[index]);
			// 计算中心
			double cx = Mom.m10 / Mom.m00;
			double cy = Mom.m01 / Mom.m00;
			printf(" cx=%f,cy=%f,Index=%d", cx, cy, contours.size() - index);
			std::cout << std::endl;
			printf("距离中线横向距离是：%f,面积是：%f", cx - 320, contourArea(contours[index]));
			std::cout << std::endl;
			cv::Point point;//特征点，用以画在图像中  
			point.x = cx;//特征点在图像中横坐标  
			point.y = cy;//特征点在图像中纵坐标  
			cv::circle(src, point, 4, cv::Scalar(0, 255, 0), -1, 8);//在图像中画出特征点，2是圆的半径 

			imshow("result", src);
			waitKey(20);
			cout << "计算中心程序执行过程耗时：" << clock() - start_svm << "毫秒" << endl;
		}
		///*imwrite("result.png", show);*/
		//if (waitKey(20) >= 0)
		//	stop = true;
		cout << "程序结束，总程序执行过程耗时：" << clock() - start_t << "毫秒" << endl;
		//waitKey(20);
		imshow("result", src);
		///*imwrite("result.png", show);*/
		if (waitKey(20) >= 0)
			stop = true;
		//waitKey();
	}
	return 0;
}