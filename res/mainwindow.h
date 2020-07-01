#pragma once
#include<QGridLayout>
#include<QtCore>
#include <QtWidgets/QMainWindow>
#include "ui_mainwindow.h"
#include<QFileDialog>
#include"sup.h"
#include <ctime>
#include <iostream>
#include<Windows.h>
#include<qdebug.h>
#include<qtime>
#include<fstream>

class mainwindow : public QMainWindow
{
	Q_OBJECT

public:
	mainwindow(QWidget *parent = Q_NULLPTR);

private:
	Ui::mainwindowClass ui;
	MpPairVec matches;
	cv::Mat image; 
	cv::Mat image_temp;
	//����˽�б��� image  
	cv::Mat image2;
	cv::Mat result;//����˽�б��� result  
private slots:    //�����źź���  
	void chooseimage1();
	void chooseimage2();
	void process1();
	void process2();
	void exit1();
};

