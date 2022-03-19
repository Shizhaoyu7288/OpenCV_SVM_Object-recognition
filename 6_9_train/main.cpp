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
int train_samples = 109;
int classes = 3;
Mat trainData;
Mat trainClasses;

//����ȫ�ֺ���

Mat readImageSaveContour(Mat src);
void getData();

Mat readImageSaveContour(Mat src)
{
	Mat imageWhite;
	Mat reverseBinaryImage;
	threshold(src, imageWhite, 100, 255, 8);
	/*imageWhite = 255 - imageWhite;*/
	bitwise_not(imageWhite, reverseBinaryImage);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(reverseBinaryImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	//�������
	/*	�Ƴ���������̵����� */
	int cmin = 30; //��С��������  
				   //int cmax = 500;    //�������  
	vector<vector<Point>>::const_iterator itc = contours.begin();
	while (itc != contours.end())
	{
		/*	if (itc->size() < cmin || itc->size() > cmax)*/
		if (itc->size() < cmin)
			itc = contours.erase(itc);
		else
			++itc;
	}
	double maxarea = 0;
	int maxAreaIdx = 0;
	for (int index = contours.size() - 1; index >= 0; index--)
	{
		double tmparea = fabs(contourArea(contours[index]));
		if (tmparea>maxarea)
		{
			maxarea = tmparea;
			maxAreaIdx = index;
		}
	}
	Rect rectangleTem = boundingRect(contours[maxAreaIdx]);
	Mat image;
	image = Mat::zeros(src.size(), CV_8UC1);
	//drawContours(image, contours, 0, Scalar(255), 2, 8, hierarchy);
	drawContours(image, contours, 0, Scalar(255), 2, 8);
	//Rect newRectangleTem(rectangleTem.x - 1, rectangleTem.y - 1, rectangleTem.width + 2, rectangleTem.height+2);
	Mat tem = image(rectangleTem);
	return tem;
}
void getData()
{
	trainData.create(train_samples*classes, sampleSize.width*sampleSize.height, CV_32FC1);
	trainClasses.create(train_samples*classes, 1, CV_32SC1);
	Mat src_image;
	char file[255];
	int i, j;
	for (i = 0; i < classes; i++)
	{
		for (j = 0; j < train_samples; j++)
		{
			//	sprintf(file, "E:/OpenCV/predictShape/predictShape/samples/s%d/%d.png", i, j);
			//ѵ��ͼ�������ļ���
			std::string path = "E:/�е���/mycode/study/study2/6_9_train/samples/t";
			char temp1[256];
			char temp2[256];
			sprintf_s(temp1, "%d", i);
			sprintf_s(temp2, "%d", j);
			path = path + temp1 + "/" + temp2 + ".jpg";
			src_image = imread(path, 0);
			if (src_image.empty())
			{
				printf("Error: Can't load image %s\n", path);
				//exit(-1);
			}
			Mat image = readImageSaveContour(src_image);
			Mat imageNewSize;
			resize(image, imageNewSize, sampleSize, CV_INTER_LINEAR);
			image.release();
			image = imageNewSize.reshape(1, 1);
			image.convertTo(trainData(Range(i*train_samples + j, i*train_samples + j + 1), Range(0, trainData.cols)), CV_32FC1);
			trainClasses.at<int>(i*train_samples + j, 0) = i;
		}
			//for (unsigned int i = 0; i <= 199; i++)//¼�븺����
			//{
			//	std::string path1 = "E:/�е���/mycode/study/study2/6_9_train/negative_samples/";
			//	/*sprintf(file, "E:/�е���/mycode/study/study2/6_9_train/negative_samples/%d.jpg", i);*/
			//	char temp3[256];
			//	sprintf_s(temp3, "%d", i);
			//	path1 = path1 + temp3 + ".jpg";
			//	Mat src_image = imread(path1, 0);
			//	if (src_image.empty())
			//	{
			//		printf("\t\t\tError: Cant load image %s\n", file);
			//		continue;
			//	}
			//	Mat image = readImageSaveContour(src_image);
			//	Mat imageNewSize;
			//	resize(image, imageNewSize, sampleSize, CV_INTER_LINEAR); //ͳһѵ�����ߴ�
			//	image.release();                       //��TemparyImage�ľ�����Ϣ�ͷţ������
			//	image = imageNewSize.reshape(1, 1);     //ͼ����Ȳ��䣬��ͼƬ����תΪһ�д���
			//	image.convertTo(trainData(Range(999 + i, 999 + i + 1), Range(0, trainData.cols)), CV_32FC1);//RangeΪ����ROI�ķ�ʽ֮һ����ʾ����ʼ����ֹ��������������ֹ����������������
			//	//trainClasses.at<int>(999 + i, 0) = -1;
			//}
		}
	}


int main() {
	int start = clock();
	getData();
	Ptr<SVM> model = SVM::create();//��ʼ����SVMģ�͵�ѵ��
	model->setType(SVM::C_SVC);
	/*model->setKernel(SVM::POLY);*/
	//model->setKernel(SVM::POLY);
	//model->setDegree(1.0);
	//model->setGamma(0.01);;
	//model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
	model->setKernel(SVM::POLY);
	model->setDegree(1.0);
	model->setGamma(0.01);;
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
	model->train(trainData, ROW_SAMPLE, trainClasses);
	model->save("E:/�е���/mycode/study/study2/6_9_train/SVM_TRAIN_DATA.xml");
	cout << "ѵ�����ˣ��ܳ���ִ�й��̺�ʱ��" << (clock() - start)/1000 << "��" << endl;
	printf("\t\t\t\tSVMѵ����������,�ܳ���ִ�й��̺�ʱ��%f��\n", (clock() - start) / 1000);
	waitKey();

	return 0;
}