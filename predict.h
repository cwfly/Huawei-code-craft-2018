#ifndef __ROUTE_H__
#define __ROUTE_H__

#include "lib_io.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <string>

using namespace std;

#define INNODE 6 //输入层节点数
#define OUTNODE 1 //输出层节点数

// 输入文件虚拟机信息—名称、CPU数和内存大小
typedef struct flavorInfo{
    string f; int cpuSize; int memSize; int cnt;
}flavorInfo;

typedef struct flavorAlloction{
    vector<string> f;
    vector<int> num_f;
}flavorAlloction;

// -----输入层节点-----
typedef struct inputNode
{
	double value; // 输入值
	vector<double> weight, wDeltaSum; //weight: 面对第一层隐含层每个节点的权值；wDeltaSum: 面对第一层隐含层每个节点权值的delta值累积
}inputNode;

// -----输出层节点-----
typedef struct outputNode   // 输出层节点
{
	int value;  // value: 节点当前输出值；
	double delta, rightvalue, bias, bDeltaSum; // delta:  输出值与正确值之间的delta值；rightvalue:  正确输出值； bias: 偏移量；bDeltaSum: bias的delta值的累积，每个节点一个
}outputNode;

// --- 隐含层节点。包含以下数值：---
// 1.value:     节点当前值；
// 2.delta:     BP推导出的delta值；
// 3.bias:      偏移量
// 4.bDeltaSum: bias的delta值的累积，每个节点一个
// 5.weight:    面对下一层（隐含层/输出层）每个节点都有权值；
// 6.wDeltaSum： weight的delta值的累积，面对下一层（隐含层/输出层）每个节点各自积累
typedef struct hiddenNode   // 隐含层节点
{
	double value, delta, bias, bDeltaSum;
	vector<double> weight, wDeltaSum;
}hiddenNode;

// --- 单个样本 ---
typedef struct sample
{
	vector<double> in;
	vector<int> out;
}sample;


void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename);


inline double get_Random()    // -1 ~ 1
{
	return ((2.0*(double)rand() / RAND_MAX) - 1);
}

// sigmoid激活函数
inline double sigmoid(double x)
{
	double ans = 1 / (1 + exp(-x));
	return ans;
}

// Lof异常点检测数据集
typedef struct dataLof {
	double k_distance; // K-邻近距离
	double lrd;        // 局部可达密度
	double lof;        // 局部异常因子
	double value;
	dataLof** neighbors;
	dataLof() {neighbors = NULL;};
	void dataLofinit(double v, int k) {
		k_distance = 0.0;
		lrd = 0.0;
		lof = 0.0;
		value = v;
		neighbors = new dataLof*[k];
	};
	/*~dataLof() {
	if(neighbors != NULL)
		delete []neighbors;
	};*/
}dataLof;

// Lof算法类
class Lof {
public:
    Lof(){};
	//~Lof() { delete []dataSet; };
	~Lof() {};
	void initLof(int k, double threshold, int length);
	void add_data(vector<double> flavorData, int length);
	//void print_lof();
	vector<int> find_outlier();
	void cal_neighbors();
	void cal_lrd();
	void cal_lof();
private:
	int length;
	int k;
	dataLof* dataSet;
	double threshold;
};

class BpNet {
public:
	vector<int> hiddenLayerNode; //隐藏层节点数
	double error;                // 误差
	double learningRate;         // 学习率
	double learnRateStart;
	inputNode* inputLayer[INNODE]; //输入层
	outputNode* outputLayer[OUTNODE]; //输出层
	hiddenNode* hiddenLayer[10][15];  //隐藏层

public:
	BpNet() {};
	~BpNet();
	void initNet(vector<int> hidden, double lrStart); //初始化
	void forwardPropagationEpoc();  // 单个样本前向传播
	void backPropagationEpoc();     // 单个样本后向传播
	void training(vector<sample> sampleGroup, int epoch);// 更新 weight, bias
	void predict(vector<sample>& testGroup);                          // 神经网络预测
	void setInput(vector<double> sampleIn);     // 设置学习样本输入
	void setOutput(vector<int> sampleOut);    // 设置学习样本输出
};


#endif
