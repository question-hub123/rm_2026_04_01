#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include "Monitor.hpp"
#include <deque>
#include <unordered_map>
#include <vector>
#include <limits>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include "armor_interfaces/msg/armor_array.hpp"
#include "EKF.hpp"  

using namespace std::chrono_literals;

struct Track {
    EKF11 ekf;                  // 11 维滤波器
    int id;                     // 装甲板全局 ID（用于关联）
    int lost_count = 3;         // 连续丢失帧数
    bool is_outpost = false;    // 是否前哨站（用于噪声参数）
    rclcpp::Time last_update_time;
};

class EKFNode : public rclcpp::Node {
public:
    EKFNode() : Node("ekf_node") {
        // 参数声明
        this->declare_parameter<int>("tracking_thres", 2);
        this->declare_parameter<int>("lost_thres", 15);
        this->declare_parameter<double>("max_match_distance", 5.0);
        this->declare_parameter<double>("init_r", 0.26);
        this->declare_parameter<double>("init_l", 0.0);
        this->declare_parameter<double>("init_h", 0.0);

        tracking_thres_ = this->get_parameter("tracking_thres").as_int();
        lost_thres_     = this->get_parameter("lost_thres").as_int();
        max_match_dist_ = this->get_parameter("max_match_distance").as_double();
        init_r_ = this->get_parameter("init_r").as_double();
        init_l_ = this->get_parameter("init_l").as_double();
        init_h_ = this->get_parameter("init_h").as_double();

        // 通信接口
        sub_armor_ = this->create_subscription<armor_interfaces::msg::ArmorArray>(
            "armor_msgs", 10, std::bind(&EKFNode::armorCallback, this, std::placeholders::_1));
        pub_filtered_ = this->create_publisher<armor_interfaces::msg::ArmorArray>("armor_msgs_filtered", 10);

        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("ekf_markers", 10);

        // 可视化（可选）
        plot_monitor_.initAnglePlot();
        gui_timer_ = this->create_wall_timer(10ms, [this]() { cv::waitKey(1); });

        RCLCPP_INFO(this->get_logger(), "EKF Node (11D RotCenter) started.");
    }

private:
    // 参数
    int tracking_thres_;
    int lost_thres_;
    double max_match_dist_;
    double init_r_, init_l_, init_h_;

    // 通信
    rclcpp::Subscription<armor_interfaces::msg::ArmorArray>::SharedPtr sub_armor_;
    rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr pub_filtered_;
    rclcpp::TimerBase::SharedPtr gui_timer_;
    rclcpp::Time last_time_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    // 跟踪器池
    std::vector<Track> tracks_;
    Monitor plot_monitor_;

    // 工具：计算观测向量 z_obs 与 armor_xyz
    void prepareObservation(const armor_interfaces::msg::Armor& arm,
                            Eigen::Vector4d& z_obs,
                            Eigen::Vector3d& armor_xyz) const {
        armor_xyz = Eigen::Vector3d(arm.x, arm.y, arm.z);
        double dist = armor_xyz.norm();
        double yaw_cam = std::atan2(arm.y, arm.x);
        double pitch_cam = std::atan2(arm.z, std::sqrt(arm.x*arm.x + arm.y*arm.y));
        // 注意：armor_msg.yaw_filtered 中我们存的是装甲板自旋角 roll
        double orientation_yaw = arm.yaw;
        z_obs << yaw_cam, pitch_cam, dist, orientation_yaw;
    }

    // 为新检测创建跟踪器
    void createNewTrack(const armor_interfaces::msg::Armor& arm, const rclcpp::Time& now) {
        Track t;
        t.id = arm.id;
        t.last_update_time = now;

        Eigen::Vector4d z_obs;
        Eigen::Vector3d armor_xyz;
        prepareObservation(arm, z_obs, armor_xyz);

        // 初始协方差对角值：位置/速度/角度/角速度/几何参数
        Eigen::Matrix<double, 11, 1> P0_diag;
        P0_diag << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,   // xc,vxc...
                   0.1, 0.1,                         // yaw, vyaw
                   0.01, 0.01, 0.01;                // r, l, h

        // 假定大符/前哨站可能只有1块板，但这里默认4块板（步兵）
        int armor_num = 4;   // 可根据 id 判断，例如前哨站 base 等可设为 1
        t.ekf.init(armor_xyz, z_obs(3), armor_num, init_r_, init_l_, init_h_, P0_diag);

        // 提取旋转中心 (xc, yc, zc)
        Eigen::Vector3d center(t.ekf.x(0), t.ekf.x(2), t.ekf.x(4));
        // 获取车体连续偏航角
        double yaw = t.ekf.x(6);

        // 计算 (armor_xyz - center) 与 (cos(yaw), sin(yaw)) 的点积
        double dx = armor_xyz.x() - center.x();
        double dy = armor_xyz.y() - center.y();
        double dot_product = dx * std::cos(yaw) + dy * std::sin(yaw);

        /*RCLCPP_INFO(this->get_logger(),
            "Manual dot: %.3f, expected r: %.3f, diff: %.3f",
            dot_product, t.ekf.x(8), std::abs(dot_product - t.ekf.x(8)));*/
            
         
        // 标记是否为前哨站（假设 id=7 为前哨站，可根据实际协议调整）
        t.is_outpost = (arm.id == 7 || arm.id == 8);
        tracks_.push_back(t);
        RCLCPP_INFO(this->get_logger(), "New track started for ID=%d", arm.id);
    }

    void armorCallback(const armor_interfaces::msg::ArmorArray::SharedPtr msg) {
        if (msg->armors.empty()) return;

        rclcpp::Time now = msg->header.stamp;
        if (last_time_.nanoseconds() == 0) {
            last_time_ = now;
            return;
        }
        double dt = (now - last_time_).seconds();
        last_time_ = now;
        if (dt <= 0.0 || dt > 0.5) dt = 0.01;

        // ---------- 1. 对所有已存在跟踪器执行预测 ----------
        for (auto& t : tracks_) {
            t.ekf.predict(dt);
        }

        // ---------- 2. 准备当前帧所有检测的观测数据 ----------
        struct Detection {
            int id;
            Eigen::Vector4d z_obs;
            Eigen::Vector3d armor_xyz;
            armor_interfaces::msg::Armor original_msg;
        };
        std::vector<Detection> detections;
        for (const auto& arm : msg->armors) {
            Detection det;
            det.id = arm.id;
            prepareObservation(arm, det.z_obs, det.armor_xyz);
            det.original_msg = arm;
            detections.push_back(det);
        }

        // ---------- 3. 数据关联（最近邻 + 距离门限） ----------
        // 简单匹配：每个跟踪器找最近的检测，且距离小于阈值
        std::vector<bool> det_matched(detections.size(), false);
        for (auto& t : tracks_) {
            Eigen::Vector3d pred_center = t.ekf.getArmorCenter(); // 返回最后匹配 id 对应的中心
            double min_dist = std::numeric_limits<double>::max();
            int best_det = -1;

            for (size_t i = 0; i < detections.size(); ++i) 
            {
                if (det_matched[i]) continue;
                double d = (detections[i].armor_xyz - pred_center).norm();
                // 简单额外约束：id 相同优先，但距离过远也拒绝
                if (d < min_dist && d < max_match_dist_) 
                {
                    min_dist = d;
                    best_det = i;
                }
            }
            if (best_det >= 0) 
            {
                auto& det = detections[best_det];
                // 执行更新（EKF11 内部会做装甲板 id 匹配和观测更新）

                double dist_to_center = (det.armor_xyz - Eigen::Vector3d(t.ekf.x(0), t.ekf.x(2), t.ekf.x(4))).norm();
                //RCLCPP_INFO(this->get_logger(), "dist=%.3f", dist_to_center);

                t.ekf.update(det.z_obs, det.armor_xyz);
                t.lost_count = 0;
                t.last_update_time = now;
                // 如果检测到的 id 与跟踪器记录的不同，更新 id（允许切换）
                if (det.id != t.id) 
                {
                    //RCLCPP_WARN(this->get_logger(), "Track ID switched from %d to %d", t.id, det.id);
                    t.id = det.id;
                }
                det_matched[best_det] = true;
            } else {
                // 未匹配，丢失计数增加
                t.lost_count++;
            }
        }

        // ---------- 4. 未匹配的检测创建新跟踪器 ----------
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!det_matched[i]) {
                createNewTrack(detections[i].original_msg, now);
            }
        }

        // ---------- 5. 清理丢失过久的跟踪器 ----------
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
            [this](const Track& t) { return t.lost_count > lost_thres_; }),
            tracks_.end());

        // ---------- 6. 选择最优目标并发布 ----------
        armor_interfaces::msg::ArmorArray out_msg;
        out_msg.header = msg->header;

        // 选择一个已收敛且距离最近的目标作为最终瞄准目标
        double best_dist = std::numeric_limits<double>::max();
        const Track* best_track = nullptr;
        for (const auto& t : tracks_) {
            if (t.lost_count > 0) continue;          // 当前帧未匹配的不考虑
            if (t.ekf.getArmorCenter().norm() < best_dist) {
                best_dist = t.ekf.getArmorCenter().norm();
                best_track = &t;
            }
        }

        if (best_track) {
            armor_interfaces::msg::Armor aim_armor;
            Eigen::Vector3d armor_center = best_track->ekf.getArmorCenter();
            double aim_yaw = std::atan2(armor_center.y(), armor_center.x());
            double aim_pitch = std::atan2(armor_center.z(), std::sqrt(armor_center.x()*armor_center.x() + armor_center.y()*armor_center.y()));

            aim_armor.x              = best_track->ekf.x(0);
            aim_armor.y              = best_track->ekf.x(2);
            aim_armor.z              = best_track->ekf.x(4);

            aim_armor.yaw            = best_track->ekf.getYaw();      // 旋转角度
            aim_armor.yaw_filtered   = best_track->ekf.x(8);          //用这个字段传装甲板半径 r，懒得改信息包了
            aim_armor.pitch_filtered = aim_pitch;                    // 预测击打 pitch
            aim_armor.is_predict     = false;
            out_msg.armors.push_back(aim_armor);

            // 画图（如果有原始检测匹配到该跟踪器，取对应检测的原始角度）
            // 为简化，这里只画平滑值
            plot_monitor_.updateAndPlotAngles(
                aim_yaw * 180.0 / M_PI,          // 平滑后的装甲板中心 yaw
                aim_yaw * 180.0 / M_PI,          // 同上（如果你不区分 raw/filtered）
                aim_pitch * 180.0 / M_PI,
                aim_pitch * 180.0 / M_PI
            );

            RCLCPP_INFO(this->get_logger(),"Cx: %.2f, Cy: %.2f, Cz: %.2f, R: %.2f",best_track->ekf.x(0), best_track->ekf.x(2), best_track->ekf.x(4), best_track->ekf.x(8));
            RCLCPP_INFO(this->get_logger(),"Ax: %.2f, Ay: %.2f, Az: %.2f, Yaw: %.2f, V_Yaw: %.2f",armor_center.x(), armor_center.y(), armor_center.z(), aim_yaw, best_track->ekf.x(6), best_track->ekf.x(7));


            //rviz2 可视化 EKF 结果
            visualization_msgs::msg::MarkerArray marker_array;

            // 1. 画车体中心 
            visualization_msgs::msg::Marker center_marker;
            center_marker.header.frame_id = msg->header.frame_id; // 保持和 YOLO 发来的一致 ("camera")
            center_marker.header.stamp = now;
            center_marker.ns = "vehicle_center";
            center_marker.id = 0;
            center_marker.type = visualization_msgs::msg::Marker::SPHERE;
            center_marker.action = visualization_msgs::msg::Marker::ADD;
            center_marker.pose.position.x = best_track->ekf.x(0);
            center_marker.pose.position.y = best_track->ekf.x(2);
            center_marker.pose.position.z = best_track->ekf.x(4);
            center_marker.scale.x = 0.1; // 直径 15cm
            center_marker.scale.y = 0.1; // 直径 15cm
            center_marker.scale.z = 0.1; // 直径 15cm
            center_marker.color.a = 0.8;  // 透明度
            center_marker.color.r = 0.0;
            center_marker.color.g = 1.0;  // 绿色
            center_marker.color.b = 0.0;
            center_marker.lifetime = rclcpp::Duration::from_seconds(0.1); // 0.1秒后自动消失
            marker_array.markers.push_back(center_marker);

            // 2. 画 4 块装甲板的 3D 线框 (LINE_LIST)
            visualization_msgs::msg::Marker armor_marker;
            armor_marker.header = center_marker.header;
            armor_marker.ns = "armors";
            armor_marker.id = 1;
            armor_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
            armor_marker.action = visualization_msgs::msg::Marker::ADD;
            armor_marker.scale.x = 0.01;  // 线条粗细
            armor_marker.color.a = 1.0;
            armor_marker.color.r = 1.0;   // 红色
            armor_marker.color.g = 0.2;
            armor_marker.color.b = 0.2;
            armor_marker.lifetime = rclcpp::Duration::from_seconds(0.1);

            double cx = best_track->ekf.x(0);
            double cy = best_track->ekf.x(2);
            double cz = best_track->ekf.x(4);
            double yaw = best_track->ekf.x(6);
            double r = best_track->ekf.x(8);
            double ARMOR_W = 0.135;
            double ARMOR_H = 0.055;

            // 根据车体中心和向内法向推算 4 块装甲板
            for (int i = 0; i < 4; ++i) {
                double face_yaw = yaw + i * (M_PI / 2.0);
                
                // 装甲板中心 (同济模型：中心 - r * 向内法向)
                double bx = cx - r * std::cos(face_yaw);
                double by = cy - r * std::sin(face_yaw);
                double bz = cz;

                // 计算四个角点（右向向量 和 上向向量）
                double rx = -std::sin(face_yaw) * (ARMOR_W / 2.0);
                double ry =  std::cos(face_yaw) * (ARMOR_W / 2.0);
                double uz = ARMOR_H / 2.0;

                geometry_msgs::msg::Point p1, p2, p3, p4;
                p1.x = bx - rx; p1.y = by - ry; p1.z = bz + uz; // 左上
                p2.x = bx + rx; p2.y = by + ry; p2.z = bz + uz; // 右上
                p3.x = bx + rx; p3.y = by + ry; p3.z = bz - uz; // 右下
                p4.x = bx - rx; p4.y = by - ry; p4.z = bz - uz; // 左下

                // 把四条边压入 LINE_LIST (每两个点连成一条线)
                auto add_line = [&](const geometry_msgs::msg::Point& a, const geometry_msgs::msg::Point& b) {
                    armor_marker.points.push_back(a);
                    armor_marker.points.push_back(b);
                };
                add_line(p1, p2); // 上边
                add_line(p2, p3); // 右边
                add_line(p3, p4); // 下边
                add_line(p4, p1); // 左边

                visualization_msgs::msg::Marker text_marker;
                text_marker.header = center_marker.header;
                text_marker.ns = "armor_ids";
                text_marker.id = 2 + i; // 给每个文字一个唯一的 ID (0给了中心球，1给了线框)
                text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING; // 面向视角的文字
                text_marker.action = visualization_msgs::msg::Marker::ADD;
                
                // 把文字放在装甲板正上方 8cm 的位置，避免和线框重叠
                text_marker.pose.position.x = bx;
                text_marker.pose.position.y = by;
                text_marker.pose.position.z = bz + 0.08; 
                
                text_marker.scale.z = 0.08; // 字体大小：8cm
                
                // 白色字体，完全不透明
                text_marker.color.a = 1.0;
                text_marker.color.r = 1.0;  
                text_marker.color.g = 1.0;
                text_marker.color.b = 1.0;
                
                // 显示数字 0, 1, 2, 3 (代表 EKF 里的第几块板)
                text_marker.text = std::to_string(i); 
                text_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
                marker_array.markers.push_back(text_marker);
            }
            marker_array.markers.push_back(armor_marker);

            // 发布 Marker
            marker_pub_->publish(marker_array);
        }

        pub_filtered_->publish(out_msg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKFNode>());
    rclcpp::shutdown();
    return 0;
}