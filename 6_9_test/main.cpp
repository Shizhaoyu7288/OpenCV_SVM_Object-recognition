
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

//ȫ�ֱ���
Size sampleSize(160, 160);//�����Ĵ�С
int train_samples = 96;
int classes = 2;
Mat trainData;
Mat trainClasses;

//typedef struct {
//	int number;
//	string name[3] = {"ball","cube","cone"};
//	float position[10][2];
//}person_t;

//����ȫ�ֺ���

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
//	//�������
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
//			//ѵ��ͼ�������ļ���
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
	Ptr<SVM> model = Algorithm::load<SVM>("E:/�е���/mycode/study/study2/6_9_train/SVM_TRAIN_DATA.xml");
	cout << "��ʼ����svm����ִ�й��̺�ʱ��" << clock() - start_svm << "����" << endl;
	//Mat src = imread("E:/�е���/mycode/study/study2/6_9_test/test/5.jpg");
	VideoCapture cap(0);//������ͷ
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
		//src = imread("E:/�е���/mycode/study/study2/6_9_test/test/5.jpg");
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
		cvtColor(src, show, CV_BGR2GRAY);//�ҶȻ�
		cout << "�Ҷȳ���ִ�й��̺�ʱ��" << clock() - start_svm1 << "����" << endl;
		int start_svm2 = clock();
		GaussianBlur(show, show, Size(7, 7), 1.5, 1.5);//��˹�˲�
		cout << "��˹�˲�����ִ�й��̺�ʱ��" << clock() - start_svm2 << "����" << endl;
		Mat imageWhite;
		start_svm = clock();
		threshold(show, imageWhite, 0, 255, CV_THRESH_BINARY);
		cout << "��ֵ������ִ�й��̺�ʱ��" << clock() - start_svm << "����" << endl;
		bitwise_not(imageWhite, imageWhite);
		/*imageWhite = 255 - imageWhite;*/
		//namedWindow("result", CV_WINDOW_AUTOSIZE);
		//imshow("result", imageWhite);
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		start_svm = clock();
		findContours(imageWhite, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		cout << "Ѱ�ұ߽����ִ�й��̺�ʱ��" << clock() - start_svm << "����" << endl;
		/*	�Ƴ���������̵����� */
		start_svm = clock();
		int cmin = 30; //��С��������  
					   //int cmax = 500;    //�������  
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
		vector<vector<Point> >::iterator itr;  //����������
		itr = contours.begin();             //ʹ�õ�����ȥ����������
		while (itr != contours.end())
		{
			area = contourArea(*itr);       //����������
			if (area<minarea)               //ɾ����С��������� 
			{
				itr = contours.erase(itr);  //itrһ��erase����Ҫ���¸�ֵ
			}
			else
			{
				itr++;
			}
			if (area>maxarea)              //Ѱ���������
			{
				maxarea = area;
			}
		}
		/// ��ÿ���ҵ���������������б�ı߽�����Բ
		vector<RotatedRect> minRect(contours.size());
		/* vector<RotatedRect> minEllipse( contours.size() );*/

		for (int i = 0; i < contours.size(); i++)
		{
			minRect[i] = minAreaRect(Mat(contours[i]));
			/*if( contours[i].size() > 5 )
			{ minEllipse[i] = fitEllipse( Mat(contours[i]) ); }*/
		}

		/// ��������������б�ı߽��ͱ߽���Բ
		/*Mat drawing = Mat::zeros( result_erase.size(), CV_8UC3 );*/
		for (int i = 0; i< contours.size(); i++)//������ɫ�߽��
		{
			Point2f rect_points[4]; minRect[i].points(rect_points);
			for (int j = 0; j < 4; j++)
				line(src, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 255, 0), 2, CV_AA);
		}
		cout << "�����Ч�߽����ִ�й��̺�ʱ��" << clock() - start_svm << "����" << endl;
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
			image.convertTo(image, CV_32FC1);//ͼ���һ��
			//Ptr<SVM> model = Algorithm::load<SVM>("E:/�е���/mycode/study/study2/6_9_train/SVM_TRAIN_DATA.xml");
			//cout << "svm����ִ�й��̺�ʱ��" << clock() - start_svm << "����" << endl;
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
				cout << "�ҵ�С�����ִ�й��̺�ʱ��" << clock() - start_svm << "����" << endl;
			}
			else if (response == 1)
			{
				cout << "    cube" << endl;
				string str = " cube," + (string)number;
				putText(src, str, Point(rectangleTem.x + rectangleTem.width / 2, rectangleTem.y + rectangleTem.height / 2),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
				cout << "�ҵ��������ִ�й��̺�ʱ��" << clock() - start_svm << "����" << endl;
			}
			else if (response == 2)
			{
				cout << "    cone" << endl;
				string str = " cone," + (string)number;
				putText(src, str, Point(rectangleTem.x + rectangleTem.width / 2, rectangleTem.y + rectangleTem.height / 2),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
				cout << "�ҵ�Բ׶����ִ�й��̺�ʱ��" << clock() - start_svm << "����" << endl;
			}
			start_svm = clock();
			cv::Moments Mom;
			Mom = cv::moments(contours[index]);
			// ��������
			double cx = Mom.m10 / Mom.m00;
			double cy = Mom.m01 / Mom.m00;
			printf(" cx=%f,cy=%f,Index=%d", cx, cy, contours.size() - index);
			std::cout << std::endl;
			printf("�������ߺ�������ǣ�%f,����ǣ�%f", cx - 320, contourArea(contours[index]));
			std::cout << std::endl;
			cv::Point point;//�����㣬���Ի���ͼ����  
			point.x = cx;//��������ͼ���к�����  
			point.y = cy;//��������ͼ����������  
			cv::circle(src, point, 4, cv::Scalar(0, 255, 0), -1, 8);//��ͼ���л��������㣬2��Բ�İ뾶 

			imshow("result", src);
			waitKey(20);
			cout << "�������ĳ���ִ�й��̺�ʱ��" << clock() - start_svm << "����" << endl;
		}
		///*imwrite("result.png", show);*/
		//if (waitKey(20) >= 0)
		//	stop = true;
		cout << "����������ܳ���ִ�й��̺�ʱ��" << clock() - start_t << "����" << endl;
		//waitKey(20);
		imshow("result", src);
		///*imwrite("result.png", show);*/
		if (waitKey(20) >= 0)
			stop = true;
		//waitKey();
	}
	return 0;
}