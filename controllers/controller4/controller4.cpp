#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <vector>
#include <webots/Camera.hpp>
#include <webots/GPS.hpp>
#include <webots/InertialUnit.hpp>
#include <webots/Keyboard.hpp>
#include <webots/Lidar.hpp>
#include <webots/Motor.hpp>
#include <webots/Robot.hpp>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace webots;
using namespace cv;

/* PRM */
#define MAPWIDTH 500  //地图宽
#define MAPHEIGHT 500 //地图高
#define SAMPLE 200    //PRM采样点数
#define FRE 60        //每多久规划一次（单位，1/60秒）
#define SAFE_D 2

#define ANT 10
#define PI 3.14159265358979323846
#define W 10.0 //TODO
#define H 10.0 //TODO
const double v = 37.0;
// PID
double P = 0, I = 0, D = 0, PID = 0;
int total_time_step = 0;
double Kp = 25, Ki = 0.03, Kd = 0.16;
//   double Kp = 14, Ki = 0.02, Kd = 0.12;
double oldP = 0, maxS = v, maxV = 0;
double medS, left_sv, right_sv;

class Record {
  public:
    double dis;
    Point last;
    Record() {
        dis = -1;
        last.x = -1;
        last.y = -1;
    }
};

class HpItem {
  public:
    double dis;
    Point now;
    Point last;
    HpItem(double dis, Point now, Point last) : dis(dis), now(now), last(last) {}
};

struct cmp //运算符重载<
{
    bool operator()(HpItem &i1, HpItem &i2) const { return i1.dis > i2.dis; }
};

double calDis(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

bool againstWall(Mat &img, Point index, int dis) {
    int x = index.x;
    int y = index.y;
    for (int i = -dis; i <= dis; i++) {
        for (int j = -dis; j <= dis; j++) {
            int x_new = x + i;
            int y_new = y + j;
            if (x_new >= 0 && x_new < img.rows) {
                if (y_new >= 0 && y_new < img.cols) {
                    if ((int) img.at<uchar>(x_new, y_new) != 255) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

bool exist(vector<Point> &V, Point index) {
    for (Point i : V) {
        if (i.x == index.x && i.y == index.y) {
            return true;
        }
    }
    return false;
}

bool passable(Mat &img, Point i1, Point i2) {
    int x1 = i1.x;
    int y1 = i1.y;
    int x2 = i2.x;
    int y2 = i2.y;
    if (abs(x1 - x2) >= abs(y1 - y2)) {
        if (x1 > x2) {
            swap(x1, x2);
            swap(y1, y2);
        }

        double k = ((double) y1 - (double) y2) / ((double) x1 - (double) x2);
        double b = (double) y1 - k * (double) x1;

        for (int x = x1 + 1; x < x2; x++) {
            int y = k * x + b;
            if ((int) img.at<uchar>(x, y) != 255) {
                return false;
            }
            if (againstWall(img, Point(x, y), SAFE_D)) {
                return false;
            }
        }
    } else {
        if (y1 > y2) {
            swap(x1, x2);
            swap(y1, y2);
        }

        double k = ((double) x1 - (double) x2) / ((double) y1 - (double) y2);
        double b = (double) x1 - k * (double) y1;

        for (int y = y1 + 1; y < y2; y++) {
            int x = k * y + b;
            if ((int) img.at<uchar>(x, y) != 255) {
                return false;
            }
            if (againstWall(img, Point(x, y), SAFE_D)) {
                return false;
            }
        }
    }
    return true;
}
vector<Point> E[MAPHEIGHT][MAPWIDTH];
Record records[MAPHEIGHT][MAPWIDTH];
vector<Point> PRM(Mat &img, Point now, Point goal, int dis_wall = 15) {
    int loopCnt = 3;
    while (loopCnt--) {
        for (int i = 0; i < MAPHEIGHT; i++) {
            for (int j = 0; j < MAPWIDTH; j++) {
                E[i][j].clear();
                records[i][j] = Record();
            }
        }
        const int rows = img.rows;
        const int cols = img.cols;
        int n = SAMPLE;
        vector<Point> V;
        V.push_back(now);
        V.push_back(goal);
        for (int i = 0; i < n; i++) {
            Point newPoint;
            while (true) {
                int x = rand() % rows;
                int y = rand() % cols;
                int pixel = img.at<uchar>(x, y);
                if (pixel == 255 && !againstWall(img, Point(x, y), dis_wall) &&
                    !exist(V, Point(x, y))) {
                    newPoint.x = x;
                    newPoint.y = y;
                    break;
                }
            }
            V.push_back(newPoint);
        }
        for (int i = 0; i < V.size(); i++) {
            for (int j = i + 1; j < V.size(); j++) {
                Point p1 = V[i];
                Point p2 = V[j];
                if (calDis(p1, p2) > 200) continue;
                if (passable(img, p1, p2)) {
                    E[p1.x][p1.y].push_back(p2);
                    E[p2.x][p2.y].push_back(p1);
                }
            }
        }
        priority_queue<HpItem, vector<HpItem>, cmp> queue;
        queue.push(HpItem(0, now, Point(-1, -1)));
        bool judge = false;
        while (!queue.empty()) {
            HpItem item = queue.top();
            queue.pop();
            double dis = item.dis;
            Point index = item.now;
            Point last = item.last;
            if (records[index.x][index.y].dis == -1) {
                records[index.x][index.y].dis = dis;
                records[index.x][index.y].last = last;
                if (index == goal) {
                    judge = true;
                    break;
                }
                last = index;
                for (Point to : E[index.x][index.y]) {
                    double new_dis = calDis(index, to) + dis;
                    queue.push(HpItem(new_dis, to, last));
                }
            }
        }
        if (!judge) continue;
        Point index = goal;
        vector<Point> path;
        while (index != Point(-1, -1)) {
            path.push_back(index);
            index = records[index.x][index.y].last;
        }
        for (int i = 0; i < path.size() - 1; i++) {
            Point p1 = path[i];
            Point p2 = path[i + 1];
            line(img, Point(p1.y, p1.x), Point(p2.y, p2.x), Scalar(0));
        }
        path.pop_back();
        reverse(path.begin(), path.end());
        return path;
    }
    return vector<Point>();
}
/* PRM */

double speedForward[4] = {v, v, v, v};
double speedBackward[4] = {-v, -v, -v, -v};
double speedLeftward[4] = {0.4 * v, 0.4 * v, v, v};
double speedRightward[4] = {v, v, 0.4 * v, 0.4 * v};
double speedLeftCircle[4] = {-0.5 * v, -0.5 * v, 0.5 * v, 0.5 * v};
double speedRightCircle[4] = {0.5 * v, 0.5 * v, -0.5 * v, -0.5 * v};

void setSpeed(int keyValue, double *speed) {
    if (keyValue == 'W')
        for (int i = 0; i < 4; ++i)
            speed[i] = speedForward[i];
    else if (keyValue == 'S')
        for (int i = 0; i < 4; ++i)
            speed[i] = speedBackward[i];
    else if (keyValue == 'A')
        for (int i = 0; i < 4; ++i)
            speed[i] = speedLeftward[i];
    else if (keyValue == 'D')
        for (int i = 0; i < 4; ++i)
            speed[i] = speedRightward[i];
    else if (keyValue == 'Q')
        for (int i = 0; i < 4; ++i)
            speed[i] = speedLeftCircle[i];
    else if (keyValue == 'E')
        for (int i = 0; i < 4; ++i)
            speed[i] = speedRightCircle[i];
    else
        for (int i = 0; i < 4; ++i)
            speed[i] = 0;
}

double dis(double x, double y, double tx, double ty) {
    return sqrt((tx - x) * (tx - x) + (ty - y) * (ty - y));
}

char judge(const double *fpos, const double *bpos, double x, double y) {
    x = W / 2 - (x * (W / MAPWIDTH));
    y = (y * (H / MAPHEIGHT)) - H / 2;

    double fx = fpos[0], fy = fpos[1];
    double bx = bpos[0], by = bpos[1];
    double k, tk;
    if (fx == bx)
        k = 10000;
    else
        k = (fy - by) / (fx - bx);
    if (x == fx)
        k = 10000;
    else
        tk = (y - fy) / (x - fx);
    double theta = atan((tk - k) / (1 + tk * k));
    double angle = 180 * theta / PI;
    printf("the angle is %.2lf\n", 180 * theta / PI);

    P = angle;
    I += P * total_time_step / 1000;
    D = D * 0.5 + (P - oldP) / total_time_step * 1000;
    PID = Kp * P + Ki * I + Kd * D;
    oldP = P;
    medS = maxS - abs(PID);
    left_sv = medS + PID;
    right_sv = medS - PID;

    if (fabs(angle) < ANT) // 小于ANT度
        if (dis(fx, fy, x, y) < dis(bx, by, x, y))
            return 'F';
        else
            return 'B';
    else if (angle < 0)
        return 'R';
    else
        return 'L';
}

void go(GPS *gps, GPS *fgps, GPS *bgps, double tx, double ty, double speed[],
        Motor *motors[]) {
    double tmpx, tmpy;
    tmpx = tx;
    tmpy = ty;
    tx = MAPWIDTH - tmpy;
    ty = MAPHEIGHT - tmpx;
    const double *pos = gps->getValues();
    const double *fpos = fgps->getValues();
    const double *bpos = bgps->getValues();
    char choice = judge(fpos, bpos, tx, ty);
    if (choice == 'F') {
        cout << "forward\n";
        setSpeed('W', speed);
    } else if (choice == 'L') {
        cout << "left\n";
        setSpeed('Q', speed);
    } else if (choice == 'R') {
        cout << "right\n";
        setSpeed('E', speed);
    } else if (choice == 'B') {
        cout << "back\n";
        setSpeed('S', speed);
    }
    // 设置电机速度
    for (int i = 0; i < 4; i++) {
        motors[i]->setVelocity(speed[i]);
    }
}

void stop(Motor *motors[]) {
    for (int i = 0; i < 4; i++) {
        motors[i]->setVelocity(0);
    }
}

// param
const int mapHeight = MAPHEIGHT;
const int mapWidth = MAPWIDTH;
const double worldHeight = H;
const double worldWidth = W;
const double world2pixel = mapHeight / worldHeight;
const int outlierCnt = 3; // 离群点检测范围

void probe(Mat &map, Lidar *lidar, int lidarRes, double carAngle, double carX,
           double carY) {
    const float *lidarImage = lidar->getRangeImage();
    double mapPointAngle, mapPointX, mapPointY;

    // 离群点判断
    for (size_t i = outlierCnt; i < lidarRes - outlierCnt; ++i) {
        if (isfinite(lidarImage[i]) && lidarImage[i] != 0) {
            double outlierCheck = 0.0;
            for (int k = i - outlierCnt; k < i + outlierCnt; ++k) {
                outlierCheck += abs(lidarImage[i] - lidarImage[k]); //计算平均距离差
            }
            outlierCheck /= (2 * outlierCnt);
            if (outlierCheck > 0.1 * lidarImage[i]) // 离群点，跳过。
                continue;

            mapPointAngle =
                carAngle + M_PI -
                double(i) / double(lidarRes) * 2.0 * M_PI;  // 计算点云角度
            mapPointX = lidarImage[i] * cos(mapPointAngle); // 计算坐标，单位m
            mapPointY = lidarImage[i] * sin(mapPointAngle);
            mapPointX += carX; // 加上平移，从小车坐标系变换到地图坐标系，单位m
            mapPointY += carY;

            int imgX = int((mapPointX + worldWidth / 2.0) *
                           world2pixel); // 地图坐标系变换到像素坐标系，单位像素
            int imgY = mapHeight - int((mapPointY + worldHeight / 2.0) * world2pixel);
            if (imgX >= 0 && imgX < map.cols && imgY >= 0 && imgY < map.rows) {
                circle(map, Point(imgX, imgY), 1, 0); // 在地图上显示雷达点云
            }
        }
    }
}

int main(int argc, char **argv) {
    Robot *robot = new Robot();
    int timeStep = (int) robot->getBasicTimeStep();

    Keyboard keyboard;
    keyboard.enable(1);

    //控制
    Motor *motors[4];
    char wheels_names[4][8] = {"motor1", "motor2", "motor3", "motor4"};
    double speed[4];
    for (int i = 0; i < 4; i++) {
        motors[i] = robot->getMotor(wheels_names[i]);
        motors[i]->setPosition(std::numeric_limits<double>::infinity());
        motors[i]->setVelocity(0.0);
        speed[i] = 0;
    }
    GPS *gps = robot->getGPS("car_gps");
    gps->enable(timeStep);
    GPS *fgps = robot->getGPS("front_gps");
    fgps->enable(timeStep);
    GPS *bgps = robot->getGPS("back_gps");
    bgps->enable(timeStep);

    const double *pos; // (x,y,z)

    InertialUnit *imu = robot->getInertialUnit("inertial unit");
    imu->enable(timeStep);
    const double *rpy; // (roll,pitch,yaw)

    Lidar *lidar = robot->getLidar("lidar");
    lidar->enable(timeStep);
    int lidarRes =
        lidar->getHorizontalResolution(); // number of lidar points per line

    Mat map(mapHeight, mapWidth, CV_8U, 255);
    int cnt = 0, maxCnt = FRE, detect = 0;
    vector<Point> path;
    int path_index = 0;
    Point destination((int) MAPHEIGHT * 0.95, (int) MAPWIDTH * 0.05);
    Point nextPoint = destination;
    Point lastPoint(-1, -1);
    int stopCnt = 0;
    int backCnt = 0;
    while (robot->step(timeStep) != -1) {
        pos = gps->getValues();
        double carX = pos[0];
        double carY = pos[1];
        int carPixelX = int((carX + worldWidth / 2.0) * world2pixel);
        int carPixelY = mapHeight - int((carY + worldHeight / 2.0) * world2pixel);
        Point nowPoint(carPixelY, carPixelX);

        rpy = imu->getRollPitchYaw();
        double carAngle = rpy[0]; // rpy[0] = roll, 但旋转小车时 yaw 不发生变化。

        // 建图
        if (cnt < 4) // 减速过程
            stop(motors);
        if (cnt == 4) {
            probe(map, lidar, lidarRes, carAngle, carX, carY);
        }

        //规划
        if (cnt == 5) {
            Mat img = map.clone();
            path = PRM(img, nowPoint, destination);
            if (path.empty()) {
                map = Mat(mapHeight, mapWidth, CV_8U, 255);
                probe(map, lidar, lidarRes, carAngle, carX, carY);
                img = map.clone();
                path = PRM(img, nowPoint, destination);
            }
            path_index = 0;
            nextPoint = path[0];
            imshow("map", img);
            waitKey(1);
        }

        //控制
        if (cnt > 5) {
            if (backCnt) {
                cout << "后退" << endl;
                backCnt--;
                double fgpsX = fgps->getValues()[0];
                double fgpsY = fgps->getValues()[1];
                double bgpsX = bgps->getValues()[0];
                double bgpsY = bgps->getValues()[1];
                double nx = nextPoint.y / world2pixel - worldWidth / 2.0;
                double ny = (mapHeight - nextPoint.x) / world2pixel - worldHeight / 2.0;
                if (dis(fgpsX, fgpsY, nx, ny) < dis(bgpsX, bgpsY, nx, ny)) {
                    setSpeed('S', speed);
                } else {
                    setSpeed('W', speed);
                }
                for (int i = 0; i < 4; i++) {
                    motors[i]->setVelocity(speed[i]);
                }
                continue;
            }
            if (calDis(nowPoint, lastPoint) < 2) {
                stopCnt++;
            } else {
                stopCnt = 0;
            }
            if (stopCnt > 30) {
                cout << "撞墙！" << endl;
                backCnt = 10;
                stopCnt = 0;
            }
            if (calDis(nowPoint, nextPoint) < 20) {
                path_index++;
                if (path_index == path.size()) {
                    stop(motors);
                    break;
                }
                nextPoint = path[path_index];
            }
            go(gps, fgps, bgps, nextPoint.x, nextPoint.y, speed, motors);
            //   cout << "当前位置：" << endl;
            //   cout << nowPoint << endl;
            //   cout << "下一位置：" << endl;
            //   cout << nextPoint << endl;

            lastPoint = nowPoint;
        }

        if (calDis(nowPoint, destination) < 10) {
            stop(motors);
            break;
        }

        cnt++;
        if (cnt >= maxCnt) cnt -= maxCnt;
    }

    delete robot;
    return 0;
}
