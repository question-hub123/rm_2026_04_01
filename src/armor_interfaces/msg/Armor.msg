# 单个装甲板信息
geometry_msgs/Point[] corners    # 四个角点的像素坐标 (顺序: 左上, 右上, 右下, 左下)
geometry_msgs/Point position      # 相机坐标系下的平移向量 tvec (x,y,z) 单位 mm
geometry_msgs/Vector3 rotation    # 旋转向量 rvec (rx,ry,rz)
geometry_msgs/Point center_2d

int32 id
float64 yaw                        # 偏航角 (度)
float64 pitch                      # 俯仰角 (度)

float64 yaw_filtered
float64 pitch_filtered