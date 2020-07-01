#include "mainwindow.h"

mainwindow::mainwindow(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	this->setWindowTitle("SURF-Image Matching System");
	this->resize(1600, 800);
}
void mainwindow::chooseimage1()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("open iamge"),//对话框名称
		".",//默认打开文件位置“.”文件目录"/"根目录
		tr("image files(*.jpg *.png *.bmp)"));//筛选器
	image = cv::imread(fileName.toLatin1().data(), CV_LOAD_IMAGE_COLOR);

	cv::cvtColor(image, image_temp, CV_BGR2RGB);//图像在QT显示前，必须转化成QImage格式
	QImage img = QImage((const unsigned char*)(image_temp.data),
		image_temp.cols, image_temp.rows, QImage::Format_RGB888);
	QPixmap Qpmap = QPixmap::fromImage(img);
	QPixmap fitpixmap = Qpmap.scaled(ui.label_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
	ui.label_2->setPixmap(fitpixmap);
	ui.label_2->setAlignment(Qt::AlignCenter);
	ui.label_2->show();

	
}
void mainwindow::chooseimage2()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("open iamge"),//对话框名称
		".",//默认打开文件位置“.”文件目录"/"根目录
		tr("image files(*.jpg *.png *.bmp)"));//筛选器
	image2 = cv::imread(fileName.toLatin1().data(), CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(image2, image_temp, CV_BGR2RGB);//图像在QT显示前，必须转化成QImage格式
											//cv::cvtColor(image2, image, CV_BGR2RGB);
	QImage img = QImage((const unsigned char*)(image_temp.data),
		image_temp.cols, image_temp.rows, QImage::Format_RGB888);
	QPixmap Qpmap = QPixmap::fromImage(img);
	QPixmap fitpixmap = Qpmap.scaled(ui.label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
	ui.label->setPixmap(fitpixmap);
	ui.label->setAlignment(Qt::AlignCenter);
	ui.label->show();
}
void mainwindow::process1()
{	
	MpVec ipts1, ipts2;
	cv::Mat img1=image.clone();
	cv::Mat img2 = image2.clone();
	clock_t start_time = clock();
	
	surfDetDes(1,&img1, ipts1, 4, 4, 2, 0.0001f);
	surfDetDes(2,&img2, ipts2, 4, 4, 2, 0.0001f);
	clock_t end_time = clock();	
	getMatches(ipts1, ipts2, matches);
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		drawPoint1(img1, matches[i].first);
		drawPoint1(img2, matches[i].second);
		const int &w = image.cols;
		cv::line(img1, cv::Point(matches[i].first.x, matches[i].first.y), cv::Point(matches[i].second.x + w, matches[i].second.y), cvScalar(255, 255, 255), 1);
		cv::line(img2, cv::Point(matches[i].first.x - w, matches[i].first.y), cv::Point(matches[i].second.x, matches[i].second.y), cvScalar(255, 255, 255), 1);
	}
	double time = (double)(end_time - start_time) / 3600;
	//cv::imshow("img1", img1);
	//cv::imshow("img2", img2);
	cv::cvtColor(img1, image_temp, CV_BGR2RGB);//图像在QT显示前，必须转化成QImage格式
												//cv::cvtColor(image2, image, CV_BGR2RGB);
	QImage img11 = QImage((const unsigned char*)(image_temp.data),
		image_temp.cols, image_temp.rows, QImage::Format_RGB888);
	QPixmap Qpmap1 = QPixmap::fromImage(img11);
	QPixmap fitpixmap = Qpmap1.scaled(ui.label_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
	ui.label_2->setPixmap(fitpixmap);
	float ratio = 255.0 / 164.0;
	float ratio2 = (float)image.rows / (float)image.cols;
	if (ratio2 > ratio)
	{
		ui.label_2->setAlignment(Qt::AlignRight);
	}
	else
	{
		ui.label_2->setAlignment(Qt::AlignVCenter);
	}
	
	cv::cvtColor(img2, image_temp, CV_BGR2RGB);//图像在QT显示前，必须转化成QImage格式
												 //cv::cvtColor(image2, image, CV_BGR2RGB);
	QImage img22 = QImage((const unsigned char*)(image_temp.data),
		image_temp.cols, image_temp.rows, QImage::Format_RGB888);
	QPixmap Qpmap2 = QPixmap::fromImage(img22);
	QPixmap fitpixmap2 = Qpmap2.scaled(ui.label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
	ui.label->setPixmap(fitpixmap2);
	float ratio3= (float)image2.rows / (float)image2.cols;
	if (ratio3 > ratio)
	{
		ui.label->setAlignment(Qt::AlignLeft);
	}
	else
	{
		ui.label->setAlignment(Qt::AlignVCenter);
	}
	ui.label->show();
	ui.label_2->show();
	std::ofstream OutFile("Test.txt", ios::app);
	OutFile << "图一特征点数目："<< ipts1.size()<< "  图二特征点数目：" << ipts2.size()<<"  匹配数：" 
		<<matches.size()<<"  时间："<<time << "s" << endl;
	OutFile.close();

}
void mainwindow::process2()
{
	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;
	for (int i = 0; i < (int)matches.size(); i++)
	{
		//采用“左图向拼接图像中添加的方法”，因此左边的是scene,右边的是obj
		scene.push_back(cv::Point2f(matches[i].first.x, matches[i].first.y));
		obj.push_back(cv::Point2f(matches[i].second.x, matches[i].second.y));
	}
	cv::Mat H = cv::findHomography(obj, scene, CV_RANSAC);
	/*H.at<float>(0,0)= 0.595;
	H.at<float>(0, 1) = 0.595;
	H.at<float>(0, 2) = 0.0279;
	H.at<float>(1, 0) = 183.121;
	H.at<float>(1, 1) = -0.216;
	H.at<float>(1, 2) = 9.253;
	H.at<float>(2, 0) = -0.00129;
	H.at<float>(2, 1) = 0.000114;
	H.at<float>(2, 2) = 1;*/
	cv::Mat resultback; //保存的是新帧经过单应矩阵变换以后的图像
	cv::warpPerspective(image2, result, H, cv::Size(2 * image2.cols, image2.rows));
	result.copyTo(resultback);
	cv::Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows));
	image.copyTo(half);
	cv::Mat result_linerblend = result.clone();
	double dblend = 0.0;
	int ioffset = image2.cols - 100;
	for (int i = 0; i<100; i++)
	{
		result_linerblend.col(ioffset + i) = result.col(ioffset + i)*(1 - dblend) + resultback.col(ioffset + i)*dblend;
		dblend = dblend + 0.01;
	}
	cv::cvtColor(result_linerblend, image_temp, CV_BGR2RGB);//图像在QT显示前，必须转化成QImage格式
												//cv::cvtColor(image2, image, CV_BGR2RGB);
	QImage img11 = QImage((const unsigned char*)(image_temp.data),
		image_temp.cols, image_temp.rows, QImage::Format_RGB888);
	QPixmap Qpmap = QPixmap::fromImage(img11);
	QPixmap fitpixmap = Qpmap.scaled(ui.label_3->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
	ui.label_3->setPixmap(fitpixmap);
	ui.label_3->setAlignment(Qt::AlignCenter);
	ui.label_3->show();
}
void mainwindow::exit1()
{
	
	QApplication::exit();
}