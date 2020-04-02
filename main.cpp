#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>
using namespace std;
using namespace cv;
void computeReprojectionErrors(const vector< vector< Point3f > >& objectPoints,
                                 const vector< vector< Point2f > >& imagePoints,
                                 const vector< Mat >& rvecs, const vector< Mat >& tvecs,
                                 const Mat& cameraMatrix , const Mat& distCoeffs) {
    vector< Point2f > imagePoints2;
    for (int i = 0; i < (int)objectPoints.size(); ++i) {
        double err;
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
                      distCoeffs, imagePoints2);
        err =(double) norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2)/objectPoints[i].size();
        cout<<"第"<<i+1<<"图片的误差为：　"<<err<<endl;
    }
}
vector< vector< Point3f > > object_points;
vector< vector< Point2f > > image_points;
int main() {
    // 水平和垂直方向内部角点的数量
    int board_width=6;
    int board_height=9;
    //图片的数量
    int num_imgs=29;
    string base_path="../calib_imgs/1";
    Size board_size=Size(board_width,board_height);

    vector<Point2f> corners;
    std::vector<cv::Point3f> objectCorners;

    // 处理所有视角
    Mat img,gray;
    for (int k = 1; k <= num_imgs; k++)
    {
        char filename[100];
        sprintf(filename,"%s/left%d.jpg",base_path.c_str(),k);
        img=imread(filename,CV_LOAD_IMAGE_COLOR);
        cvtColor(img,gray,CV_BGR2GRAY);

        bool found=false;
        found=findChessboardCorners(img,board_size,corners,CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        if(found)
        {
            cornerSubPix(gray,corners,cv::Size(5, 5), cv::Size(-1, -1),TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(img,board_size,corners,found);
        }
        // 场景中的三维点:
        // 在棋盘坐标系中,初始化棋盘中的角点
        // 角点的三维坐标(X,Y,Z)= (j,i,0)
        // 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
        for (int i=0; i<board_size.height; i++) {
            for (int j=0; j<board_size.width; j++) {
                objectCorners.push_back(cv::Point3f(j, i, 0.0f));
//                cout<<"("<<j<<","<<i<<")"<<endl;
            }
        }
        if (found) {
            cout << k << ". Found corners!" << endl;
            image_points.push_back(corners);
            object_points.push_back(objectCorners);
            objectCorners.clear();
        }
//        imshow("1",img);
//        waitKey(0);
    }

    cv::Mat instrisincMatrix=cv::Mat::eye(3,3,CV_64F);
    cv::Mat distortionCoeff=cv::Mat::zeros(4,1,CV_64F);
    vector<Mat> rvecs, tvecs;
    int flag = 0;
    flag |= CV_CALIB_FIX_K4;
    flag |= CV_CALIB_FIX_K5;
    calibrateCamera(object_points,image_points,img.size(),instrisincMatrix,distortionCoeff,rvecs,tvecs);

    cout<<"instrisincMatrix: "<<endl<<instrisincMatrix<<endl;
    cout<<"distortionCoeff: "<<endl<<distortionCoeff<<endl;
    computeReprojectionErrors(object_points, image_points, rvecs, tvecs, instrisincMatrix, distortionCoeff);
    //通过畸变校正效果查看摄像机标定效果
    cv::Mat R = cv::Mat::eye(3, 3, CV_32FC1);
    cv::Mat mapx, mapy, undistortImg;
    cv::initUndistortRectifyMap(instrisincMatrix, distortionCoeff, R, instrisincMatrix, img.size(), CV_32FC1, mapx, mapy);
    cv::remap(img, undistortImg, mapx, mapy, CV_INTER_LINEAR);
    cv::imshow("undistortImg", undistortImg);
    cv::waitKey(0);


    return 0;
}