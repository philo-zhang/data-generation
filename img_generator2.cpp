#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<opencv/cv.hpp>
#include<opencv/highgui.h>

#define PI 3.1415926 

using namespace std;
using namespace cv;

int main(){
	string source = "../../voc/";
    string dest = "../../cnn_affine_data/";
	string path;
	char imgName[50];
	char outImgName1[50];
	char outImgName2[50];
    ifstream fread((source+"labels.txt").c_str());
    if(!fread.is_open())
        printf("Unable to open the input file!");
	
	ofstream fwrite((dest + "labels.txt").c_str());
	if(!fwrite.is_open())
		printf("Unable to open the output file!");
    float temp[4];
	float rect[4];
	Mat affMat, rotatedSrc, imgObjTemp, imgObjOri, imgObjTrans;
	Mat shiftArray(1, 4, CV_32F, Scalar::all(0));
	Mat randM(1, 4, CV_32F, Scalar::all(0));
	Mat img;
	Mat newArray(1, 4, CV_32F, Scalar::all(0));
	vector<int> params;
	params.push_back(CV_IMWRITE_JPEG_QUALITY);
	params.push_back(100);
	int count = 0;
    int imgNum = 11540;
	for(int i=0;i<1;++i){
		for(int j = 0;j<imgNum;++j){
			sprintf(imgName,"%simgs/%05d.jpg", source.c_str(), j+1);
			fread>>temp[0]>>temp[1]>>temp[2]>>temp[3];
            rect[0] = temp[0];
            rect[1] = temp[1];
            rect[2] = temp[2]-temp[0];
            rect[3] = temp[3]-temp[1];
			Point2f center(rect[0]+rect[2]/2, rect[1]+rect[3]/2);
			Size imgSize(rect[2], rect[3]);
			Size objSize(64,64);
			float ratio[] = {rect[2]/objSize.width, rect[3]/objSize.height};
			cout<<imgName<<endl;
			img = imread(imgName,1);
            if(rect[2]*rect[3]>img.rows*img.cols/4)
                continue;
            if(rect[2]<objSize.width || rect[3]<objSize.height)
                continue;
			Rect objRect(rect[0], rect[1], rect[2], rect[3]);
			imgObjTemp = img(objRect).clone();
			resize(imgObjTemp, imgObjOri, objSize); 
			for (int k = 0; k<100; ++k){
				randu(randM, Scalar::all(-1), Scalar::all(1));
				shiftArray.at<float>(0,0) = rect[2]/3;
				shiftArray.at<float>(0,1) = rect[3]/3;
				shiftArray.at<float>(0,2) = 15;
				shiftArray.at<float>(0,3) = 0.2;
				newArray = shiftArray.mul(randM);
				Point2f deltaCenter(newArray.at<float>(0,0), newArray.at<float>(0,1));
				float deltaAngle = newArray.at<float>(0,2);
				float deltaScale = newArray.at<float>(0,3);
				affMat = getRotationMatrix2D(center+deltaCenter,deltaAngle,1.0);
				warpAffine(img,rotatedSrc,affMat,Size(img.cols,img.rows),INTER_CUBIC);
				getRectSubPix(rotatedSrc,imgSize+Size(rect[2]*deltaScale,rect[3]*deltaScale),center+deltaCenter,imgObjTemp);
				resize(imgObjTemp, imgObjTrans, objSize);
				sprintf(outImgName1, "%s/data1/%06d.jpg", dest.c_str(), count);
				sprintf(outImgName2, "%s/data2/%06d.jpg", dest.c_str(), count++);
                if(imgObjOri.channels()!=3){
                    cvtColor(imgObjOri, imgObjOri, CV_GRAY2RGB);
                    cvtColor(imgObjTrans, imgObjTrans, CV_GRAY2RGB);
                }
				imwrite(outImgName1, imgObjOri, params);
				imwrite(outImgName2, imgObjTrans, params);
                float scale = 1+deltaScale;
                float cosine = cos(deltaAngle*PI/180);
                float sine = sin(deltaAngle*PI/180);
                float m[3][3] = {{scale*cosine, -scale*sine, deltaCenter.x},{scale*sine, scale*cosine, deltaCenter.y}, {0, 0, 1}};
                Mat affineM = Mat(3, 3, CV_32F, m);
                float oriCoor[3][4] = {{-rect[2]/2, rect[2]/2, rect[2]/2, -rect[2]/2}, {-rect[3]/2, -rect[3]/2, rect[3]/2, rect[3]/2}, {1, 1, 1, 1}};
                Mat oriRect = Mat(3, 4, CV_32F, oriCoor);
                Mat newRect = affineM*oriRect;
                Mat disRect = newRect-oriRect;
                /*cout<<"deltas:"<<deltaScale<<" "<<deltaAngle<<" "<<deltaCenter.x<<" "<<deltaCenter.y<<endl;
                cout<<"scale, cosine, sine:"<<scale<<" "<<cosine<<" "<<sine<<endl;
                cout<<"affineM:"<<endl;
                for(int y=0; y<3; ++y){
                    for(int z=0; z<3; ++z)
                        cout<<affineM.at<float>(y,z)<<" ";
                    cout<<endl;
                }
                cout<<"oriRect:"<<endl;
                for(int y=0; y<3; ++y){
                    for(int z=0; z<4; ++z)
                        cout<<oriRect.at<float>(y,z)<<" ";
                    cout<<endl;
                }
                cout<<"newRect:"<<endl;
                for(int y=0; y<3; ++y){
                    for(int z=0; z<4; ++z)
                        cout<<newRect.at<float>(y,z)<<" ";
                    cout<<endl;
                }*/
                for(int x=0; x<4;++x){
                    fwrite<<disRect.at<float>(0,x)/ratio[0]<<" "<<disRect.at<float>(1,x)/ratio[1];
                    if(x<3)
                        fwrite<<" ";
                }
                fwrite<<endl;
			}
		}
		fread.close();
	}
	fwrite.close();
	return 0;
}
