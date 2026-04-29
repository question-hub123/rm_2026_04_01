int32 id
float64 x                          # PnP解算得到的相机系x (米)
float64 y                          # PnP解算得到的相机系y (米)
float64 z                          # PnP解算得到的相机系z (米)
float64 yaw                        # 偏航角 (弧度)
float64 pitch                      # 俯仰角 (弧度)

float64 yaw_filtered
float64 pitch_filtered

bool is_predict