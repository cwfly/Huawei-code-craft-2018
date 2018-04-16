#include "predict.h"
#include <stdio.h>
#include<sstream>
#include<algorithm>
#include<cstring>
#include<limits>

// class Lof 初始化
void Lof::initLof(int k, double threshold, int length) {
	this->k = k;
	this->threshold = threshold;
	this->length = length;
	dataSet = new dataLof[length]();
}

// 添加数据
void Lof::add_data(vector<double> flavorData, int length) {
	for (int i = 0; i < length; i++)
		dataSet[i].dataLofinit(flavorData[i], k);

}

void Lof::cal_neighbors() {
	for (int i = 0; i < length; i++) {
		double distances[k];
		for (int n = 0; n < k; n++)
			distances[n] = numeric_limits<double>::max();
		for (int j = 0; j < length; j++) {
			if (i == j)
				continue;
			double dist = (dataSet[i].value - dataSet[j].value)*(dataSet[i].value - dataSet[j].value);
			dist = sqrt(dist);
			for (int h = 0; h < k; h++) {
				if (distances[h] > dist) {
					for (int p = k - 1; p > h; p--) {
						distances[p] = distances[p - 1];
						dataSet[i].neighbors[p] = dataSet[i].neighbors[p - 1];
					}
					distances[h] = dist;
					dataSet[i].neighbors[h] = &(dataSet[j]);
					break;

				}
			}

		}
		dataSet[i].k_distance = distances[k - 1];
	}
}

void Lof::cal_lrd() {
	for (int i = 0; i < length; i++) {
		double reach_dist = 0.0001;
		for (int j = 0; j < k; j++) {
			double d = (dataSet[i].value - dataSet[j].value)*(dataSet[i].value - dataSet[j].value);
			d = sqrt(d);
			reach_dist += (d > dataSet[i].neighbors[j]->k_distance) ? d : dataSet[i].neighbors[j]->k_distance;
		}
		dataSet[i].lrd = (double)k / reach_dist;
	}
}

void Lof::cal_lof() {
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < k; j++)
			dataSet[i].lof += dataSet[i].neighbors[j]->lrd;
		dataSet[i].lof = dataSet[i].lof / dataSet[i].lrd;
		dataSet[i].lof /= double(k);
	}
}
/**
void Lof::print_lof() {
	for (int i = 0; i < length; i++)
		cout << dataSet[i].lof << endl;
}
**/
vector<int> Lof::find_outlier() {
	cal_neighbors();
	cal_lrd();
	cal_lof();
	//print_lof();
	vector<int> index;
	for (int i = 0; i < length; i++) {

		if (dataSet[i].lof > threshold) {
			index.push_back(i);

		}
	}

	for(int i=0;i<length;i++)
        delete []dataSet[i].neighbors;
    delete []dataSet;
	return index;
}

// 初始化各层weight、bias
void BpNet::initNet(vector<int> hidden, double lrStart) {
    hiddenLayerNode.assign(hidden.begin(), hidden.end());
	srand((unsigned)time(NULL));        // 随机数种子
	error = 100.0;// error初始值
	learnRateStart = lrStart;
	learningRate = lrStart; //初始化学习率
	int hideLayer = hiddenLayerNode.size(); // 隐藏层数

	// 初始化输入层
	for (int i = 0; i < INNODE; i++) {
		inputLayer[i] = new inputNode();
		for (int j = 0; j < hiddenLayerNode[0]; j++) {
			inputLayer[i]->weight.push_back(get_Random());
			inputLayer[i]->wDeltaSum.push_back(0.0);
		}
	}

	// 初始化隐藏层
	for (int i = 0; i < hideLayer; i++) {
		if (i == hideLayer - 1) {
			for (int j = 0; j < hiddenLayerNode[i]; j++) {
				hiddenLayer[i][j] = new hiddenNode();
				hiddenLayer[i][j]->bias = get_Random();
				for (int k = 0; k < OUTNODE; k++) {
					hiddenLayer[i][j]->weight.push_back(get_Random());
					hiddenLayer[i][j]->wDeltaSum.push_back(0.0);
					//cout<<"init: "<<hiddenLayer[i][j]->wDeltaSum[k]<<endl;
				}
			}
		}
		else {
			for (int j = 0; j < hiddenLayerNode[i]; j++) {
				hiddenLayer[i][j] = new hiddenNode();
				hiddenLayer[i][j]->bias = get_Random();
				for (int k = 0; k < hiddenLayerNode[i + 1]; k++) {
					hiddenLayer[i][j]->weight.push_back(get_Random());
					hiddenLayer[i][j]->wDeltaSum.push_back(0.0);
				}

			}
		}
	}

	// 初始化输出层
	for (int i = 0; i < OUTNODE; i++)
	{
		outputLayer[i] = new outputNode();
		outputLayer[i]->bias = get_Random();
	}
}

void BpNet::forwardPropagationEpoc() {
	// 前向传播——隐藏层
	int hideLayerNum=hiddenLayerNode.size();
	//cout<<"hideLayerNum: "<<hideLayerNum<<endl;
	for (int i = 0; i < hideLayerNum; i++) {
	    //cout<<i<<": "<<endl;
		if (i == 0) {
			for (int j = 0; j < hiddenLayerNode[i]; j++) {
			    //cout<<"ij: "<<i<<j<<": "<<endl;
                double sum = 0.0;
				for (int k = 0; k < INNODE; k++) {
				    //cout<<"ijk: "<<i<<j<<k<<": "<<endl;
					sum += inputLayer[k]->value*inputLayer[k]->weight[j];
				}
				sum += hiddenLayer[i][j]->bias;
				hiddenLayer[i][j]->value = sigmoid(sum);
			}
		}
		else {
			for (int j = 0; j < hiddenLayerNode[i]; j++) {
			    //cout<<"ij: "<<i<<j<<": "<<endl;
				double sum = 0.0;
				for (int k = 0; k < hiddenLayerNode[i - 1]; k++) {
				    //cout<<"ijk: "<<i<<j<<k<<": "<<endl;
					sum += hiddenLayer[i - 1][k]->value*hiddenLayer[i - 1][k]->weight[j];
				}
				sum += hiddenLayer[i][j]->bias;
				hiddenLayer[i][j]->value = sigmoid(sum);
			}
		}
	}

	// 前向传播——输出层
	for (int i = 0; i < OUTNODE; i++) {
	    //cout<<"out: "<<i<<endl;
		double sum = 0.0;
		for (int j = 0; j < hiddenLayerNode[hideLayerNum - 1]; j++) {
		   // cout<<"out: "<<i<<j<<endl;
			sum += hiddenLayer[hideLayerNum - 1][j]->value*hiddenLayer[hideLayerNum - 1][j]->weight[i];
		}
		sum += outputLayer[i]->bias;
		outputLayer[i]->value = sum; // 回归预测，输出层无激活函数
	}
}

void BpNet::backPropagationEpoc() {
    int hideLayerNum=hiddenLayerNode.size();
	// 反向传播——输出层
	// 计算delta

	for (int i = 0; i < OUTNODE; i++) {
	   // cout<<"backout: "<<i<<endl;
		double temp = fabs(outputLayer[i]->value-outputLayer[i]->rightvalue);
		//cout<<"backout: "<<i<<"1: "<<endl;
		error += temp * temp / 2;
		//cout<<"backout: "<<i<<"2: "<<endl;
		outputLayer[i]->delta = (outputLayer[i]->value - outputLayer[i]->rightvalue);
		//cout<<"backout: "<<i<<"3: "<<endl;
	}

	// 反向传播——隐藏层
	// 计算delta
	//cout<<"backhidden: "<<"1: "<<endl;
	for (int i = hideLayerNum - 1; i >= 0; i--) {
	   // cout<<"backhidden: "<<i<<": "<<endl;
		if (i == hideLayerNum - 1) {
			for (int j = 0; j < hiddenLayerNode[i]; j++)
			{
			    //cout<<"backhidden: "<<i<<j<<": "<<endl;
				double sum = 0.0;
				for (int k = 0; k < OUTNODE; k++) {
				   // cout<<"backhidden: "<<i<<j<<k<<": "<<endl;
					sum += outputLayer[k]->delta*hiddenLayer[i][j]->weight[k];
                   // cout<<"backhidden: "<<i<<j<<k<<"1: "<<endl;
				}
				//cout<<"backhidden: "<<i<<j<<"0: "<<endl;
				hiddenLayer[i][j]->delta = sum * (1 - hiddenLayer[i][j]->value)*(hiddenLayer[i][j]->value);
				//cout<<"backhidden: "<<i<<j<<"1: "<<endl;
			}
		}
		else {
			for (int j = 0; j < hiddenLayerNode[i]; j++) {
			    //cout<<"backhidden: "<<i<<j<<": "<<endl;
				double sum = 0.0;
				for (int k = 0; k < hiddenLayerNode[i+1]; k++) {
				   // cout<<"backhidden: "<<i<<j<<k<<": "<<endl;
					sum += hiddenLayer[i+1][k]->delta*hiddenLayer[i][j]->weight[k];
					//cout<<"backhidden: "<<i<<j<<k<<"1: "<<endl;
				}
				//cout<<"backhidden: "<<i<<j<<"0: "<<endl;
				hiddenLayer[i][j]->delta = sum * (1 - hiddenLayer[i][j]->value)*(hiddenLayer[i][j]->value);
				//cout<<"backhidden: "<<i<<j<<"1: "<<endl;
			}
		}
	}

	// 反向传播——输入层
	// 更新delta sum
	for (int i = 0; i < INNODE; i++) {
	   // cout<<"innode: "<<i<<endl;
		for (int j = 0; j < hiddenLayerNode[0]; j++) {
		    //cout<<"innode: "<<i<<j<<endl;
			inputLayer[i]->wDeltaSum[j] += inputLayer[i]->value * hiddenLayer[0][j]->delta;
			//cout<<"innode: "<<i<<j<<"5: "<<endl;
		}
	}
	// 反向传播——隐藏层
	// 更新delta sum 和 bias delta sum
	for (int i = 0; i < hideLayerNum; i++) {
	   // cout<<"hedden updata: "<<i<<endl;
		if (i == hideLayerNum-1) {
			for (int j = 0; j < hiddenLayerNode[i]; j++) {
			    //cout<<"hedden updata: "<<i<<j<<endl;
				hiddenLayer[i][j]->bDeltaSum += hiddenLayer[i][j]->delta;
				//cout<<"hedden updata: "<<i<<j<<"o: "<<endl;
				for (int k = 0; k < OUTNODE; k++) {
				   // cout<<"hedden updata: "<<i<<j<<k<<endl;
					hiddenLayer[i][j]->wDeltaSum[k] += hiddenLayer[i][j]->value * outputLayer[k]->delta;
					//cout<<"hedden updata: "<<i<<j<<k<<"1: "<<endl;
				}
			}
		}
		else {
			for (int j = 0; j < hiddenLayerNode[i]; j++) {
			   // cout<<"hedden updata: "<<i<<j<<endl;
				hiddenLayer[i][j]->bDeltaSum += hiddenLayer[i][j]->delta;
				//cout<<"hedden updata: "<<i<<j<<"a: "<<endl;
				for (int k = 0; k < hiddenLayerNode[i + 1]; k++) {
				 //   cout<<"hedden updata: "<<i<<j<<k<<endl;
				    //cout<<"hiddenLayer[i][j]->value: "<<hiddenLayer[i][j]->value<<endl;
				    //cout<<"a"<<endl;
				    //cout<<"hiddenLayer[i + 1][k]->delta: "<<hiddenLayer[i + 1][k]->delta<<endl;
				   // cout<<"b"<<endl;
				    //cout<<"hiddenLayer[i][j]->wDeltaSum[k]: "<<hiddenLayer[i][j]->wDeltaSum[k]<<endl;
					hiddenLayer[i][j]->wDeltaSum[k] += hiddenLayer[i][j]->value * hiddenLayer[i + 1][k]->delta;
					//cout<<"hedden updata: "<<i<<j<<k<<"1: "<<endl;
				}
			}
		}
	}

	// 反向传播——输出层
	// 更新bias delta sum
	for (int i = 0; i < OUTNODE; i++)
		{
		  // cout<<"out updata: "<<i<<endl;
		   outputLayer[i]->bDeltaSum += outputLayer[i]->delta;
          //cout<<"out updata: "<<i<<"1: "<<endl;
		}
}

void BpNet::training(vector<sample> sampleGroup, int epoch) {
	int sampleNum = sampleGroup.size();
	int hideLayerNum=hiddenLayerNode.size();
    int count=0;
    double decayRate=1;
    // adam优化参数

	double vDwIn[15] = { 0 };
	double sDwIn[15] = { 0 };
	double vDbH[10][15] = { 0 };
	double sDbH[10][15] = { 0 };
	double vDwH[8][15][15] = { 0 };
	double sDwH[8][15][15] = { 0 };
	double vDbO[2] = { 0 };
	double sDbO[2] = { 0 };
	double b1 = 0.9, b2 = 0.999;
	double b3 = 0.000000001;


	while (epoch--)
	{



        //cout<<"error: "<<error<<endl;
		count++;
        if ((count % 500) == 0)
		{
			learningRate = learnRateStart * 1 / (1 + decayRate * count);

		}

		// initialize delta sum
		for (int i = 0; i < INNODE; i++)
			inputLayer[i]->wDeltaSum.assign(inputLayer[i]->wDeltaSum.size(), 0.0);

		for (int i = 0; i < hideLayerNum; i++) {
			for (int j = 0; j < hiddenLayerNode[i]; j++)
			{
				hiddenLayer[i][j]->wDeltaSum.assign(hiddenLayer[i][j]->wDeltaSum.size(), 0.0);
				hiddenLayer[i][j]->bDeltaSum = 0.0;
			}
		}
		for (int i = 0; i < OUTNODE; i++)
			outputLayer[i]->bDeltaSum = 0.0;
		for (int iter = 0; iter < sampleNum; iter++)
		{
			setInput(sampleGroup[iter].in);
			setOutput(sampleGroup[iter].out);

			forwardPropagationEpoc();
			backPropagationEpoc();

		}

		// 反向传播——输入层
		// 更新weight

		for (int i = 0; i < INNODE; i++) {

			for (int j = 0; j < hiddenLayerNode[0]; j++) {
                vDwIn[j] = b1 * vDwIn[j] + (1 - b1)*inputLayer[i]->wDeltaSum[j] / sampleNum;
                sDwIn[j] = b2 * sDwIn[j] + (1 - b2)*inputLayer[i]->wDeltaSum[j] / sampleNum*inputLayer[i]->wDeltaSum[j] / sampleNum;
				inputLayer[i]->weight[j] -= learningRate * vDwIn[j] / (sqrt(sDwIn[j]) + b3);
				//inputLayer[i]->weight[j] -= learningRate * inputLayer[i]->wDeltaSum[j] / sampleNum;

			}
		}

		// 反向传播——隐藏层
		// 更新weight和bias
		for (int i = 0; i < hideLayerNum; i++) {
			if (i == hideLayerNum - 1) {
				for (int j = 0; j < hiddenLayerNode[i]; j++) {

					// bias
					vDbH[i][j] = b1 * vDbH[i][j] + (1 - b1)*hiddenLayer[i][j]->bDeltaSum / sampleNum;
					sDbH[i][j] = b2 * sDbH[i][j] + (1 - b2)*hiddenLayer[i][j]->bDeltaSum / sampleNum*hiddenLayer[i][j]->bDeltaSum / sampleNum;
					hiddenLayer[i][j]->bias -= learningRate * vDbH[i][j] / (sqrt(sDbH[i][j]) + b3);
					//hiddenLayer[i][j]->bias -= learningRate * hiddenLayer[i][j]->bDeltaSum / sampleNum;
					// weight
					for (int k = 0; k < OUTNODE; k++) {
					    vDwH[i][j][k] = b1 * vDwH[i][j][k] + (1 - b1)*hiddenLayer[i][j]->bDeltaSum / sampleNum;
					    sDwH[i][j][k] = b2 * sDwH[i][j][k] + (1 - b2)*hiddenLayer[i][j]->bDeltaSum / sampleNum*hiddenLayer[i][j]->bDeltaSum / sampleNum;
						hiddenLayer[i][j]->weight[k] -= learningRate * vDwH[i][j][k] / (sqrt(sDwH[i][j][k]) + b3);
						//hiddenLayer[i][j]->weight[k] -= learningRate * hiddenLayer[i][j]->wDeltaSum[k] / sampleNum;
					}
				}
			}
			else {
				for (int j = 0; j < hiddenLayerNode[i]; j++) {

					// bias
					vDbH[i][j] = b1 * vDbH[i][j] + (1 - b1)*hiddenLayer[i][j]->bDeltaSum / sampleNum;
					sDbH[i][j] = b2 * sDbH[i][j] + (1 - b2)*hiddenLayer[i][j]->bDeltaSum / sampleNum*hiddenLayer[i][j]->bDeltaSum / sampleNum;
					hiddenLayer[i][j]->bias -= learningRate * vDbH[i][j] / (sqrt(sDbH[i][j]) + b3);
					//hiddenLayer[i][j]->bias -= learningRate * hiddenLayer[i][j]->bDeltaSum / sampleNum;

					// weight
						for (int k = 0; k < hiddenLayerNode[i+1]; k++)
						{
						    vDwH[i][j][k] = b1 * vDwH[i][j][k] + (1 - b1)*hiddenLayer[i][j]->wDeltaSum[k] / sampleNum;
						    sDwH[i][j][k] = b2 * sDwH[i][j][k] + (1 - b2)*hiddenLayer[i][j]->wDeltaSum[k] / sampleNum*hiddenLayer[i][j]->wDeltaSum[k] / sampleNum;
							hiddenLayer[i][j]->weight[k] -= learningRate * vDwH[i][j][k] / (sqrt(sDwH[i][j][k]) + b3);
							//hiddenLayer[i][j]->weight[k] -= learningRate * hiddenLayer[i][j]->wDeltaSum[k] / sampleNum;
						}

				}
			}
		}

		// 反向传播——输出
		// 更新bias
		for (int i = 0; i < OUTNODE; i++) {
		    vDbO[i] = b1 * vDbO[i] + (1 - b1)*outputLayer[i]->bDeltaSum / sampleNum;
		    sDbO[i] = b2 * sDbO[i] + (1 - b2)*outputLayer[i]->bDeltaSum / sampleNum*outputLayer[i]->bDeltaSum / sampleNum;
			outputLayer[i]->bias -= learningRate * vDbO[i] / (sqrt(sDbO[i]) + b3);
			//outputLayer[i]->bias -= learningRate * outputLayer[i]->bDeltaSum / sampleNum;
		}

	}
}

void BpNet::predict(vector<sample>& testGroup) {
	int testNum = testGroup.size();
	int hideLayerNum=hiddenLayerNode.size();

	for (int iter = 0; iter < testNum; iter++) {
		//testGroup[iter].out.clear();
		setInput(testGroup[iter].in);
        //cout<<"iter: "<<iter<<endl;
		// 前向传播——隐藏层
		for (int i = 0; i < hideLayerNum; i++) {
		    //cout<<"hiddenlayer: "<<endl;
			if (i == 0) {
				for (int j = 0; j < hiddenLayerNode[i]; j++) {
					double sum = 0.0;
					for (int k = 0; k < INNODE; k++) {

						sum += inputLayer[k]->value*inputLayer[k]->weight[j];
					}

					sum += hiddenLayer[i][j]->bias;

					hiddenLayer[i][j]->value = sigmoid(sum);

				}
			}
			else {
				for (int j = 0; j < hiddenLayerNode[i]; j++) {
					double sum = 0.0;
					for (int k = 0; k < hiddenLayerNode[i - 1]; k++) {



						sum += hiddenLayer[i - 1][k]->value*hiddenLayer[i - 1][k]->weight[j];

					}

					sum += hiddenLayer[i][j]->bias;

					hiddenLayer[i][j]->value = sigmoid(sum);

				}
			}
		}

		// 前向传播——输出层
		for (int i = 0; i < OUTNODE; i++) {
		   // cout<<"outlayer: "<<endl;
			double sum = 0.0;
			for (int j = 0; j < hiddenLayerNode[hideLayerNum - 1]; j++) {

				sum += hiddenLayer[hideLayerNum - 1][j]->value*hiddenLayer[hideLayerNum - 1][j]->weight[i];
			}

			sum += outputLayer[i]->bias;
			if(sum<0)
			    sum=0;
			outputLayer[i]->value = sum; // 回归预测，输出层无激活函数

			testGroup[iter].out.push_back(outputLayer[i]->value);
		}
	}
}

void BpNet::setInput(vector<double> sampleIn)
{
	for (int i = 0; i < INNODE; i++) inputLayer[i]->value = sampleIn[i];
}

void BpNet::setOutput(vector<int> sampleOut)
{
	for (int i = 0; i < OUTNODE; i++) outputLayer[i]->rightvalue = sampleOut[i];
}

BpNet::~BpNet() {
	for (int i = 0; i < INNODE; i++) {
		delete inputLayer[i];
	}

	int hideLayer = hiddenLayerNode.size(); // 隐藏层数
	for (int i = 0; i < hideLayer; i++) {
		for (int j = 0; j < hiddenLayerNode[i]; j++)
			delete hiddenLayer[i][j];
	}

	for (int i = 0; i < OUTNODE; i++)
		delete outputLayer[i];
}

//你要完成的功能总入口——日期的处理，计算天数差
int day_process(int Y1, int M1, int D1, int Y2, int M2, int D2) {
	int month_day[12] = { 31,28,31,30,31,30,31,31,30,31,30,31 };
	if (Y1 % 4 == 0)
		month_day[1] = 29;
	while (Y1 != Y2) {
		Y2--;
		if (Y2 % 4 == 0)
			D2 += 366;
		else
			D2 += 365;
	}
	while (M2 != M1) {
		if (M2 > M1) {
			M2--;
			D2 += month_day[M2 - 1];
		}
		else {
			D2 -= month_day[M2 - 1];
			M2++;
		}
	}
	return D2 - D1 + 1;
}


// 数据处理——只处理info中指定的 flavor 数据
/*****************

 info: 输入信息，包括需要服务器数据、需要预测的flavor规格名称、需要优化的维度（CPU or MEM）、需要预测的时间跨度
 data: 历史数据。用于训练
 data_num: data行数
 data_days: data中数据的时间跨度（即数据开始时间到结束时间之间的天数）
 flavor: 存储info中的flavor规格名称、cpu核数和内存大小
 ecs: 存储info中的物理服务器信息
 cm: 存储需要优化的维度(CPU or MEM)
 info_days: 需要预测的时间跨度

*****************/
vector<vector<double> > data_process(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, int& data_days, vector<flavorInfo>& flavor, vector<int>& ecs, string& cm, int& info_days) {
	int n = atoi(info[2]);  // 读取虚拟机规格数量
	istringstream read_info0(info[0]);
	read_info0 >> ecs[0] >> ecs[1] >> ecs[2]; //读入物理服务器的CPU核数、内存大小、硬盘大小

	// 将需要预测的flavor读入flavor数组中
	for (int i = 0; i < n; i++) {
		flavorInfo t;
		istringstream read_info(info[3 + i]);
		read_info >> t.f>>t.cpuSize>>t.memSize;
		//cout<<t.f<<' '<<t.cpuSize<<' '<<t.memSize<<endl;
		read_info.str("");
		read_info.clear();
		flavor.push_back(t);
	}

	istringstream read_info1(info[4+n]);
	read_info1 >> cm; //读取需要优化的资源维度名称(CPU or MEM)
	char c;
	int Y_1, M_1, D_1, Y_2, M_2, D_2;
	istringstream read_info2(info[6 + n]);
	read_info2 >> Y_1>> c >> M_1 >> c >> D_1; //读入预测开始时间
	istringstream read_info3(info[7 + n]);
	read_info3 >> Y_2 >> c >> M_2 >> c >> D_2; //读入预测结束时间

	info_days = day_process(Y_1, M_1, D_1, Y_2, M_2, D_2)-1; //预测的时间跨度

	// 计算data中总共有多少天的数据
	// code begin
	int Y_orgin, M_orgin, D_orgin, Y_last, M_last, D_last, m;
	string a, b;

	istringstream read_data0(data[0]);
	read_data0 >> a >> b;
	read_data0 >> Y_orgin >> c >> M_orgin >> c >> D_orgin;
	istringstream read_data1(data[data_num-1]);
	read_data1 >> a >> b;
	read_data1 >> Y_last >> c >> M_last >> c >> D_last;
	m = day_process(Y_orgin, M_orgin, D_orgin, Y_last, M_last, D_last);

	data_days = m;
	// code end
	// 有多少天的数据，flavorData则有多少列
	vector<vector<double> > flavorData(n, vector<double>(m, 0)); // 每一行代表某个需要预测的虚拟机规格在指定时间段里每天被请求的数量,flavorData[i][j]=a 表示flavor[i].f在第j天一共被请求a次
	for (int i = 0; i < data_num; i++) {
		string flavorID, flavorName;
		char tp;
		int Y, M, D, D_value;
		istringstream read_data(data[i]);
		read_data >> flavorID >> flavorName;
		//提取年月日
		read_data >> Y >> tp >> M >> tp >> D;


		//计算时间间隔（天）
		D_value = day_process(Y_orgin, M_orgin, D_orgin, Y, M, D)-1;

		//检查flavorName是否在flavor中，如果不在，不统计其数据，结束循环，如果在，flavorData对应位置加1

		for(int j=0; j<n; j++){
		    if(flavor[j].f==flavorName)
		        {
		           flavorData[j][D_value]++;
		           break;
		        }
		}


		//if (i == data_num - 1)
		//	data_days = D_value;
	}
	return flavorData;
}


// 计算 data 的均值和方差
void ave_var(double &ave, double &var, vector<double> data) {
	int len = data.size();
	double sum = 0.0;
	for (int i = 0; i < len; i++) {
		sum += data[i];
	}
	ave = sum / len;
	sum = 0.0;
	for (int i = 0; i < len; i++) {
		sum += ((data[i] - ave)*(data[i] - ave));
	}
	var = sum / len;
}

// 数据处理——将数据分为x(历史数据) 和 y(正确值)
void dataSplitX_Y(vector<vector<sample> >& trainData, vector<vector<double> > flavorData, int info_days, int n, vector<vector<double> >& maxIn, vector<vector<double> >& minIn) {
	int row = flavorData.size(); // 行数，即需要预测的虚拟机类型数目
	int col = flavorData[0].size(); // 列数，即训练数据集的数据
	bool div = false;
	int loop_num; //循环次数
				  // 以前(n-1)个每info_days数据的和、均值、方差
	if (col % (n * info_days) == 0) {
		loop_num = col / (n * info_days);
		div = true;
	}
	else {
		loop_num = col / (n * info_days) + 1;
		div = false;
	}


	double ave = 0.0, var = 0.0; // ave:均值，var:方差
								 // 第一层for循环对应着哪个虚拟机的数据
	for (int k = 0; k < row; k++) {
		int count = 0; // 循环次数
		while (count<loop_num)
		{
			vector<double> inSum; //存取每info_days个数据的和，总共n-1
			vector<double> aveData; //存取每info_days个数据的均值，总共n-1
			vector<double> varData; //存取每info_days个数据的方差，总共n-1
			int outSum = 0;
			int num = 0;
			double sum = 0.0;
			vector<double> data;
			if (div) {
				for (int i = 0; i <= n * info_days; i++) {
					if (i <= (n - 1)*info_days) { //前n-1个info_days数据作为X
						if (num < info_days) { //每info_days个数据的和为X
							num++;
							sum += flavorData[k][i + (n * info_days)*count];
							data.push_back(flavorData[k][i + (n * info_days)*count]);
						}
						else {
							inSum.push_back(sum);
							ave_var(ave, var, data);
							aveData.push_back(ave);
							varData.push_back(var);
							i--;
							data.clear();
							num = 0;
							sum = 0.0;
						}
					}
					else { //后info_days个数据和为Y

						outSum += flavorData[k][i - 1 + (n * info_days)*count];
					}
				}
			}
			else {
				if (count < loop_num - 1) {

					for (int i = 0; i <= n * info_days; i++) {
						if (i <= (n - 1)*info_days) { //前n-1个info_days数据作为X
							if (num < info_days) { //每info_days个数据的和为X
								num++;
								sum += flavorData[k][i + (n * info_days)*count];
								data.push_back(flavorData[k][i + (n * info_days)*count]);
							}
							else {
								inSum.push_back(sum);
								ave_var(ave, var, data);
								aveData.push_back(ave);
								varData.push_back(var);
								data.clear();
								i--;
								num = 0;
								sum = 0.0;
							}
						}
						else
						{

							outSum += flavorData[k][i - 1 + (n * info_days)*count];
						}
					}
				}
				// 如果col % (n * info_days)不等于0，则最后一组数据去最后n*info_days个数据
				else {
					for (int i = col - n * info_days; i <= col; i++)
						if (i <= col - info_days)
						{
							if (num < info_days) { //每info_days个数据的和为X
								num++;
								sum += flavorData[k][i];
								data.push_back(flavorData[k][i]);
							}
							else {
								inSum.push_back(sum);
								ave_var(ave, var, data);
								aveData.push_back(ave);
								varData.push_back(var);
								data.clear();
								i--;
								num = 0;
								sum = 0.0;
							}
						}
						else
							outSum += flavorData[k][i - 1];
				}
			}
			// 输入有info_days个数据的和、均值、方差组成
			// 将inSum里的值压入输入数据
			for (int j = 0; j < n - 1; j++) {
				trainData[k][count].in.push_back(inSum[j]);
			}

			// 将aveData里的值压入输入数据
			for (int j = 0; j < n - 1; j++) {
				trainData[k][count].in.push_back(aveData[j]);
			}

			// 将varData里的值压入输入数据
			for (int j = 0; j < n - 1; j++) {
				trainData[k][count].in.push_back(varData[j]);
			}

			// 压入输出数据
			trainData[k][count].out.push_back(outSum);
			count++;
		}



		// 数据归一化
		int h = trainData[0][0].in.size();
		vector<double> x;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < loop_num; j++) {
				x.push_back(trainData[k][j].in[i]);
			}
		}
		vector<double> mxIn; // 存储最大值
		vector<double> mnIn; // 存储最小值
		int p = x.size();
		for (int i = 0; i < p; i += loop_num) {
			double mx = *(max_element(x.begin() + i, x.begin() + i + loop_num));
			double mn = *(min_element(x.begin() + i, x.begin() + i + loop_num));
			mxIn.push_back(mx);
			mnIn.push_back(mn);
		}
		for (int i = 0; i < loop_num; i++) {
			for (int j = 0; j < h; j++) {
			    if(mxIn[j]!=mnIn[j])
				   trainData[k][i].in[j] = (trainData[k][i].in[j] - mnIn[j]) / (mxIn[j] - mnIn[j]);
			}
		}
		maxIn.push_back(mxIn);
		minIn.push_back(mnIn);
	}

}


// 按cpuSize从大到小排序
bool cmp_cpu(flavorInfo a, flavorInfo b)
{
		return a.cpuSize > b.cpuSize;
}

// 按memSize从大到小排序
bool cmp_mem(flavorInfo a, flavorInfo b)
{
		return a.memSize > b.memSize;
}

// 降序最佳适应算法
/**************
flavor:       存储info中的flavor规格名称、cpu核数和内存大小以及预测数
ecs:          存储info中的物理服务器信息，ecs[0]:CPU核数，ecs[1]:内存大小
cm:           存储需要优化的维度(CPU or MEM)
ecsNum：      所需要的物理服务器数量
***************/
void dp_find_solution(int number, int w_CPU, int w_MEM, int* weight_MEM, int* weight_CPU, int* number_temp, int* value, int*** f, int* push_res) {
	if (number > 0) {
		if (f[number][w_CPU][w_MEM] == f[number - 1][w_CPU][w_MEM]) {
			dp_find_solution(number - 1, w_CPU, w_MEM, weight_MEM, weight_CPU, number_temp, value, f, push_res);
		}
		else if (w_CPU - weight_CPU[number - 1] >= 0 && w_MEM - weight_MEM[number - 1] >= 0 && f[number][w_CPU][w_MEM] == f[number - 1][w_CPU - weight_CPU[number - 1]][w_MEM - weight_MEM[number - 1]] + value[number - 1]) {
			push_res[number_temp[number - 1]]++;
			dp_find_solution(number - 1, w_CPU - weight_CPU[number - 1], w_MEM - weight_MEM[number - 1], weight_MEM, weight_CPU, number_temp, value, f, push_res);
		}
	}
}

void beg_dp(int* weight_MEM, int* weight_CPU, int* number_temp, int* value, int nums, int w_MEM, int w_CPU, int* push_res) {
	int ***f = new int**[nums + 1];
	for (int i = 0; i < nums + 1; i++) {
		f[i] = new int*[w_CPU + 1];
		for (int j = 0; j <= w_CPU; j++)
			f[i][j] = new int[w_MEM + 1]();
	}
	for (int i = 1; i <= nums; i++) {
		for (int w_temp_CPU = weight_CPU[i - 1]; w_temp_CPU <= w_CPU; w_temp_CPU++) {
			for (int w_temp_MEM = weight_MEM[i - 1]; w_temp_MEM <= w_MEM; w_temp_MEM++) {
				if (f[i - 1][w_temp_CPU][w_temp_MEM] < f[i - 1][w_temp_CPU - weight_CPU[i - 1]][w_temp_MEM - weight_MEM[i - 1]] + value[i - 1])
					f[i][w_temp_CPU][w_temp_MEM] = f[i - 1][w_temp_CPU - weight_CPU[i - 1]][w_temp_MEM - weight_MEM[i - 1]] + value[i - 1];
				else
					f[i][w_temp_CPU][w_temp_MEM] = f[i - 1][w_temp_CPU][w_temp_MEM];
			}
		}
	}
	dp_find_solution(nums, w_CPU, w_MEM, weight_MEM, weight_CPU, number_temp, value, f, push_res);
	for (int i = 0; i <= nums; i++) {
		for (int j = 0; j <= w_CPU; j++)
			delete[]f[i][j];
		delete[]f[i];
	}
	delete[]f;
}

vector<flavorAlloction> bfd(vector<flavorInfo> flavor, vector<int> ecs, string cm, int& ecsNum) {
	int vir_sum_all = 0;
	int *vir_sum = new int[flavor.size()]();
	vector<flavorAlloction> res;
	for (int i = 0; i < flavor.size(); i++) {
		vir_sum_all += flavor[i].cnt;
		vir_sum[i] = flavor[i].cnt;
	}
	for (int count = 0; vir_sum_all > 0; count++) {
		flavorAlloction servr_temp;
		//处理数据
		int* weight_MEM = new int[vir_sum_all]();
		int* weight_CPU = new int[vir_sum_all]();
		int* number_temp = new int[vir_sum_all]();
		int* push_res = new int[flavor.size()]();
		int n_temp = 0;
		for (int i = 0; i < vir_sum_all;) {
			for (int j = 0; j < vir_sum[n_temp]; j++) {
				number_temp[i] = n_temp;
				weight_CPU[i] = flavor[n_temp].cpuSize;
				weight_MEM[i] = flavor[n_temp].memSize / 1024;
				i++;
			}
			n_temp++;
		}
		//放置
		if (cm == "CPU")
			beg_dp(weight_MEM, weight_CPU, number_temp, weight_CPU, vir_sum_all, ecs[1], ecs[0], push_res);
		else if (cm == "MEM")
			beg_dp(weight_MEM, weight_CPU, number_temp, weight_MEM, vir_sum_all, ecs[1], ecs[0], push_res);
		for (int i = 0; i < flavor.size(); i++) {
			if (push_res[i] != 0) {
				servr_temp.f.push_back(flavor[i].f);
				servr_temp.num_f.push_back(push_res[i]);
			}
			vir_sum[i] -= push_res[i];
			vir_sum_all -= push_res[i];
		}
		delete[]weight_MEM;
		delete[]weight_CPU;
		delete[]number_temp;
		delete[]push_res;
		ecsNum++;

		res.push_back(servr_temp);
	}
	delete[]vir_sum;
	return res;
}
//你要完成的功能总入口
void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename)
{
	int data_days=0; //data_days: data中数据的时间跨度（即数据开始时间到结束时间之间的天数）
    vector<flavorInfo> flavor; // flavor: 存储info中的flavor规格名称
    vector<int> ecs(3);    // ecs: 存储info中的物理服务器信息
    string cm;             // cm: 存储需要优化的维度(CPU or MEM)
    int info_days=0;       // info_days: 需要预测的时间跨度

    // vector<vector<int> > flavorData = data_process(info, data, data_num, data_days, flavor, ecs, cm, info_days);
    vector<vector<double> > flavorData=data_process(info, data, data_num, data_days, flavor, ecs, cm, info_days);
    int row = flavorData.size();
    int col = flavorData[0].size();

    vector<vector<int> > res(row);
    Lof check;
    for(int i=0;i<row;i++){


        int k=1;
        double threshold=1.0001;
        check.initLof(k, threshold, col);
        check.add_data(flavorData[i], col);
        res[i] = check.find_outlier();

    }

    // deal fault data
    for(int i=0;i<row;i++){
       int s = res[i].size();
       for(int j=0; j < s; j++){
           int id = res[i][j];
           if(id==0){
               flavorData[i][id] = (flavorData[i][id+1]+flavorData[i][id+2])/2.0;
           }
           else if(id==col-1){
               flavorData[i][id] = (flavorData[i][id-2]+flavorData[i][id-3])/2.0;
           }
           else{
              flavorData[i][id] = (flavorData[i][id-1]+flavorData[i][id+1])/2.0;
           }
       }
    }

    int n=3;
    int train_num;
    if (col % (n * info_days) == 0)
       train_num = col / (n * info_days);
    else
       train_num = col / (n * info_days)+1;
    sample init[train_num];
    vector<vector<sample> > trainData(row, vector<sample>(init, init+train_num));
    vector<vector<double> > maxIn;
    vector<vector<double> > minIn;
    dataSplitX_Y(trainData, flavorData, info_days, n, maxIn, minIn);
    int num1=trainData.size();

    int col1=trainData[0].size();



    //vector<int> hiddenLayerNode(4,3);
    vector<int> hidden;
    hidden.push_back(10);
    hidden.push_back(6);
    hidden.push_back(6);
    hidden.push_back(4);
    hidden.push_back(4);
    //hidden.push_back(4);
    double lr = 0.3;
    vector<sample> testData(1);

    BpNet trainNet;
    trainNet.initNet(hidden, lr);

    for(int i=0; i<row; i++){

        trainNet.training(trainData[i],10000);
        testData[0].in.clear();
        int num=0;
        double sumb=0.0;
        double var=0.0;
        double ave=0.0;
        vector<double> aData;
        vector<double> vData;
        vector<double> sumIn;
        vector<double> d;
        for(int j=col-(n-1)*info_days; j<col; j++)
           {
              if(num<info_days){
                  num++;
                  sumb+=flavorData[i][j];
                  d.push_back(flavorData[i][j]);

              }
              else{
                  sumIn.push_back(sumb);
                  ave_var(ave, var, d);
                  aData.push_back(ave);
                  vData.push_back(var);
                  d.clear();
                  num=0;
                  sumb=0.0;
                  j--;

              }

              //testData[0].in.push_back(flavorData[i][j]);
           }
           sumIn.push_back(sumb);
           ave_var(ave, var, d);
           aData.push_back(ave);
           vData.push_back(var);
           for(int h=0; h<n-1; h++)
              testData[0].in.push_back(sumIn[h]);
           for(int h=0; h<n-1; h++)
              testData[0].in.push_back(aData[h]);
           for(int h=0; h<n-1; h++)
              testData[0].in.push_back(vData[h]);

           int g = testData[0].in.size();
           for(int h=0; h<g; h++){
              if(maxIn[i][h] != minIn[i][h])
                 testData[0].in[h]=(testData[0].in[h]-minIn[i][h])/(maxIn[i][h]-minIn[i][h]);
           }


        trainNet.predict(testData);

    }

    //trainNet.~BpNet();
    //vector<int> flavorResult(row+1, 0);


    for(int i=0; i<row; i++)
        flavor[i].cnt = testData[0].out[i]; // compute sum for every flavor
    int sum=0;
    for(int i=0; i<row; i++)
        sum += flavor[i].cnt;
    //flavorResult[0] = sum;


    int ecsNum=0;
    vector<flavorAlloction> alloction_result = bfd(flavor, ecs, cm, ecsNum);

    // output
    char result_file[5000];
    char result_temp[100];
    stringstream stream;
    stream<<sum;
    stream>>result_file;
    strcat(result_file, "\n");

    for(int i=0; i<row; i++){
        stream.clear();
        stream<<flavor[i].f;
        stream>>result_temp;
        strcat(result_file, result_temp);
        strcat(result_file, " ");
        stream.clear();
        stream<<flavor[i].cnt;
        stream>>result_temp;
        strcat(result_file, result_temp);
        strcat(result_file, "\n");
    }
    strcat(result_file, "\n");
    stream.clear();
    stream<<ecsNum;
    stream>>result_temp;
    strcat(result_file, result_temp);
    strcat(result_file, "\n");
    for(int i=0; i<ecsNum; i++){
        stream.clear();
        stream<<i+1;
        stream>>result_temp;
        strcat(result_file, result_temp);
        strcat(result_file, " ");
        for(int j=0; j<alloction_result[i].f.size(); j++){

            stream.clear();
            stream<<alloction_result[i].f[j];
            stream>>result_temp;
            strcat(result_file, result_temp);
            strcat(result_file, " ");
            stream.clear();
            stream<<alloction_result[i].num_f[j];
            stream>>result_temp;
            strcat(result_file, result_temp);
            strcat(result_file, " ");
        }
        strcat(result_file, "\n");
    }


	// 直接调用输出文件的方法输出到指定文件中(ps请注意格式的正确性，如果有解，第一行只有一个数据；第二行为空；第三行开始才是具体的数据，数据之间用一个空格分隔开)
	write_result(result_file, filename);

}
