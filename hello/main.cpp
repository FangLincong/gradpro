#include "opencv2/opencv.hpp"
using namespace cv;
#include <iostream>
using namespace std;
IplImage* org = 0;
IplImage* img = 0;
IplImage* tmp = 0;
IplImage* dst = 0;
string path = "E://dst.jpg";
//对轮廓按面积降序排列  
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
	threshold(gray_img, temp_img, 100, 255, CV_THRESH_BINARY);//对图像进行二值化
	threshold(gray_img_template, temp_img_template, 100, 255, CV_THRESH_BINARY);
	
	//imwrite("目标图处理.jpg", temp_img);
	//imwrite("模版图处理.jpg", temp_img_template);
	vector<vector<Point>> contours_img, contours_template;//目标图，模版图
	findContours(temp_img, contours_img, CV_RETR_TREE, CHAIN_APPROX_NONE);//提取轮廓元素
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
	//imwrite("模版图轮廓.jpg",img_template);

	double pro = 1;//相似度，越接近0越好
	double min_pro = 999;//对应的最优匹配值
	int min_kk = -1;//对应的最优匹配的下标
	for (int kk = 0; kk < contours_img.size(); kk++)
	{
		if (contourArea(contours_img[kk]) < 10000)//面积阈值筛选
		{
			break;
		}
		rt = boundingRect(contours_img[kk]);
		if (rt.height <= rt.width)//垃圾桶是矩形
		{
			continue;
		}

		pro = matchShapes(contours_template[0], contours_img[kk], CV_CONTOURS_MATCH_I3, 1.0);//进行轮廓匹配

		if (pro < min_pro)
		{
			min_pro = pro;
			min_kk = kk;
		}

		cout << kk << "==" << pro << endl;
	}

	rt = boundingRect(contours_img[min_kk]);
	rectangle(img, rt, Scalar(20, 120, 245), 2);

	cout << "相似度最高轮廓下标：" << min_kk << endl;
	cout << "目标形心坐标：" << rt.x + rt.width / 2 << "," << rt.y + rt.height / 2 << endl;

	//imwrite("目标寻找结果.jpg", img);
	namedWindow("结果");
	imshow("结果", img);
	waitKey();
}
void matchnum(string goal, IplImage * ipl)//在goal图中找到temp
{
		//1.查找模版图像的轮廓
	
		Mat templateImg = cv::cvarrToMat(ipl);
	
		Mat copyImg1 = templateImg.clone();
		//RGB到gray
		cvtColor(templateImg, templateImg, CV_BGR2GRAY);
	
		threshold(templateImg, templateImg, 100, 255, CV_THRESH_BINARY);//确保黑中找白
		namedWindow("模版", CV_WINDOW_NORMAL);
	
		imshow("模版", templateImg);
	
		vector<vector<Point>> contours1;
	
		findContours(templateImg, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//最外层轮廓
	
		//drawContours(copyImg1, contours1, -1, Scalar(0, 255, 0), 2, 8);
		
		//2.查找待测试图像的轮廓
	
		Mat testImg = imread(goal, CV_LOAD_IMAGE_COLOR);
	
		Mat copyImg2 = testImg.clone();
	
		cvtColor(testImg, testImg, CV_BGR2GRAY);
	
		threshold(testImg, testImg, 100, 255, CV_THRESH_BINARY);//确保黑中找白
		//namedWindow("thresh2", CV_WINDOW_NORMAL);
		imshow("目标", testImg);
	
		vector<vector<Point>> contours2;
	
		findContours(testImg, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//最外层轮廓
	
		cout << contours2.size() << endl;														 //3.形状匹配---比较两个形状或轮廓间的相似度
	
		for (int i = 0; i < contours2.size(); i++)//遍历待测试图像的轮廓
	
		{
	
			//返回此轮廓与模版轮廓之间的相似度,a0越小越相似
	
			double a0 = matchShapes(contours1[0], contours2[i], CV_CONTOURS_MATCH_I3, 0.0);
	
			cout << "模版轮廓与待测试图像轮廓" << i << "的相似度:" << a0 << endl;//输出两个轮廓间的相似度
	
			if (a0<0.1)//如果此轮廓与模版轮廓的相似度小于0.1
			{
				drawContours(copyImg2, contours2, i, Scalar(250, 0, 0), 2, 8);//则在待测试图像上画出此轮廓
			}
			//namedWindow("copyImg2", CV_WINDOW_NORMAL);
			imshow("结果", copyImg2);
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