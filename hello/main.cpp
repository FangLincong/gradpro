#include "opencv2/opencv.hpp"
using namespace cv;
#include <iostream>
using namespace std;
IplImage* org = 0;
IplImage* img = 0;
IplImage* tmp = 0;
IplImage* dst = 0;
string path = "E://dst.jpg";
//�������������������  
bool biggerSort(vector<Point> v1, vector<Point> v2)
{
	return contourArea(v1)>contourArea(v2);
}
void on_mouse(int event, int x, int y, int flags, void* ustc)
{
	static CvPoint pre_pt = { -1,-1 };
	static CvPoint cur_pt = { -1,-1 };
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);
	char temp[16];

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		cvCopy(org, img);
		sprintf_s(temp, "(%d,%d)", x, y);
		pre_pt = cvPoint(x, y);
		cvPutText(img, temp, pre_pt, &font, cvScalar(0, 0, 100, 125));
		cvCircle(img, pre_pt, 3, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		cvShowImage("img", img);
		cvCopy(img, tmp);
	}
	else if (event == CV_EVENT_MOUSEMOVE && !(flags & CV_EVENT_FLAG_LBUTTON))
	{
		cvCopy(tmp, img);
		sprintf_s(temp, "(%d,%d)", x, y);
		cur_pt = cvPoint(x, y);
		cvPutText(img, temp, cur_pt, &font, cvScalar(0, 0, 100, 125));
		cvShowImage("img", img);
	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
	{
		cvCopy(tmp, img);
		sprintf_s(temp, "(%d,%d)", x, y);
		cur_pt = cvPoint(x, y);
		cvPutText(img, temp, cur_pt, &font, cvScalar(0, 0, 100, 125));
		cvRectangle(img, pre_pt, cur_pt, cvScalar(0, 255, 0, 0), 1, 8, 0);
		cvShowImage("img", img);
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		cvCopy(tmp, img);
		sprintf_s(temp, "(%d,%d)", x, y);
		cur_pt = cvPoint(x, y);
		cvPutText(img, temp, cur_pt, &font, cvScalar(0, 0, 100, 125));
		cvCircle(img, cur_pt, 3, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		cvRectangle(img, pre_pt, cur_pt, cvScalar(0, 255, 0, 0), 1, 8, 0);
		cvShowImage("img", img);
		cvCopy(img, tmp);
		int width = abs(pre_pt.x - cur_pt.x);
		int height = abs(pre_pt.y - cur_pt.y);
		if (width == 0 || height == 0)
		{
			cvDestroyWindow("dst");
			return;
		}
		dst = cvCreateImage(cvSize(width, height), org->depth, org->nChannels);
		CvRect rect;
		if (pre_pt.x<cur_pt.x && pre_pt.y<cur_pt.y)
		{
			rect = cvRect(pre_pt.x, pre_pt.y, width, height);
		}
		else if (pre_pt.x>cur_pt.x && pre_pt.y<cur_pt.y)
		{
			rect = cvRect(cur_pt.x, pre_pt.y, width, height);
		}
		else if (pre_pt.x>cur_pt.x && pre_pt.y>cur_pt.y)
		{
			rect = cvRect(cur_pt.x, cur_pt.y, width, height);
		}
		else if (pre_pt.x<cur_pt.x && pre_pt.y>cur_pt.y)
		{
			rect = cvRect(pre_pt.x, cur_pt.y, width, height);
		}
		cvSetImageROI(org, rect);
		cvCopy(org, dst);
		cvResetImageROI(org);
		cvDestroyWindow("dst");
		cvNamedWindow("dst", 1);
		cvShowImage("dst", dst);
		const char* p = path.data();
		//cvSaveImage(p, dst);
	}
}
void matchjpg(string goal, IplImage * ipl)
{
	Mat img = imread(goal, CV_LOAD_IMAGE_COLOR);
	//cv::Mat m1 = cv::cvarrToMat(ipl);
	Mat img_template = cv::cvarrToMat(ipl);
	Mat gray_img, gray_img_template;
	cvtColor(img, gray_img, COLOR_BGR2GRAY);
	cvtColor(img_template, gray_img_template, COLOR_BGR2GRAY);
	Mat temp_img, temp_img_template;
	threshold(gray_img, temp_img, 100, 255, CV_THRESH_BINARY);//��ͼ����ж�ֵ��
	threshold(gray_img_template, temp_img_template, 100, 255, CV_THRESH_BINARY);
	
	//imwrite("Ŀ��ͼ����.jpg", temp_img);
	//imwrite("ģ��ͼ����.jpg", temp_img_template);
	vector<vector<Point>> contours_img, contours_template;//Ŀ��ͼ��ģ��ͼ
	findContours(temp_img, contours_img, CV_RETR_TREE, CHAIN_APPROX_NONE);//��ȡ����Ԫ��
	findContours(temp_img_template, contours_template, CV_RETR_TREE, CHAIN_APPROX_NONE);
	cout << contours_img.size() << endl;
	cout << contours_template.size() << endl;
	//std::sort(contours_img.begin(), contours_img.end(), biggerSort);
	//std::sort(contours_template.begin(), contours_template.end(), biggerSort);
	Rect rt;
	//for (int kk = 0; kk < contours_template.size(); kk++)
	//{
	//    rt = boundingRect(contours_template[kk]);
	//    rectangle(img_template, rt, Scalar(0,0,255),2);   
	//}
	//imwrite("ģ��ͼ����.jpg",img_template);

	double pro = 1;//���ƶȣ�Խ�ӽ�0Խ��
	double min_pro = 999;//��Ӧ������ƥ��ֵ
	int min_kk = -1;//��Ӧ������ƥ����±�
	for (int kk = 0; kk < contours_img.size(); kk++)
	{
		if (contourArea(contours_img[kk]) < 10000)//�����ֵɸѡ
		{
			break;
		}
		rt = boundingRect(contours_img[kk]);
		if (rt.height <= rt.width)//����Ͱ�Ǿ���
		{
			continue;
		}

		pro = matchShapes(contours_template[0], contours_img[kk], CV_CONTOURS_MATCH_I3, 1.0);//��������ƥ��

		if (pro < min_pro)
		{
			min_pro = pro;
			min_kk = kk;
		}

		cout << kk << "==" << pro << endl;
	}

	rt = boundingRect(contours_img[min_kk]);
	rectangle(img, rt, Scalar(20, 120, 245), 2);

	cout << "���ƶ���������±꣺" << min_kk << endl;
	cout << "Ŀ���������꣺" << rt.x + rt.width / 2 << "," << rt.y + rt.height / 2 << endl;

	//imwrite("Ŀ��Ѱ�ҽ��.jpg", img);
	namedWindow("���");
	imshow("���", img);
	waitKey();
}
void matchnum(string goal, IplImage * ipl)//��goalͼ���ҵ�temp
{
		//1.����ģ��ͼ�������
	
		Mat templateImg = cv::cvarrToMat(ipl);
	
		Mat copyImg1 = templateImg.clone();
		//RGB��gray
		cvtColor(templateImg, templateImg, CV_BGR2GRAY);
	
		threshold(templateImg, templateImg, 100, 255, CV_THRESH_BINARY);//ȷ�������Ұ�
		namedWindow("ģ��", CV_WINDOW_NORMAL);
	
		imshow("ģ��", templateImg);
	
		vector<vector<Point>> contours1;
	
		findContours(templateImg, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//���������
	
		//drawContours(copyImg1, contours1, -1, Scalar(0, 255, 0), 2, 8);
		
		//2.���Ҵ�����ͼ�������
	
		Mat testImg = imread(goal, CV_LOAD_IMAGE_COLOR);
	
		Mat copyImg2 = testImg.clone();
	
		cvtColor(testImg, testImg, CV_BGR2GRAY);
	
		threshold(testImg, testImg, 100, 255, CV_THRESH_BINARY);//ȷ�������Ұ�
		//namedWindow("thresh2", CV_WINDOW_NORMAL);
		imshow("Ŀ��", testImg);
	
		vector<vector<Point>> contours2;
	
		findContours(testImg, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//���������
	
		cout << contours2.size() << endl;														 //3.��״ƥ��---�Ƚ�������״������������ƶ�
	
		for (int i = 0; i < contours2.size(); i++)//����������ͼ�������
	
		{
	
			//���ش�������ģ������֮������ƶ�,a0ԽСԽ����
	
			double a0 = matchShapes(contours1[0], contours2[i], CV_CONTOURS_MATCH_I3, 0.0);
	
			cout << "ģ�������������ͼ������" << i << "�����ƶ�:" << a0 << endl;//�����������������ƶ�
	
			if (a0<0.1)//�����������ģ�����������ƶ�С��0.1
			{
				drawContours(copyImg2, contours2, i, Scalar(250, 0, 0), 2, 8);//���ڴ�����ͼ���ϻ���������
			}
			//namedWindow("copyImg2", CV_WINDOW_NORMAL);
			imshow("���", copyImg2);
		}
		
		waitKey();
}
void sellect(string goal)
{
	// Read image
	const char* p = goal.data();
	org = cvLoadImage(p, 1);
	img = cvCloneImage(org);
	tmp = cvCloneImage(org);
	cvNamedWindow("img", 1);
	cvSetMouseCallback("img", on_mouse, 0);

	cvShowImage("img", img);
	cvWaitKey(0);
	cvDestroyAllWindows();
	cvReleaseImage(&org);
	cvReleaseImage(&img);
	cvReleaseImage(&tmp);
	//cvReleaseImage(&dst);
}
int main()
{
	string load = "../img/num.PNG";
	sellect(load);
	matchnum(load, dst);
	//matchjpg(load, dst);
	return 0;
}