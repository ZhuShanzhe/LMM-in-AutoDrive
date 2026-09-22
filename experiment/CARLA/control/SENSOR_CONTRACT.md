# 模型传感器契约

UniversalVLAController支持模型配置中的sensor_requirements：

```json
{
  "sensor_requirements": {
    "cameras": ["front", "left", "right", "rear"],
    "lidar": true
  }
}
```

声明后该配置优先于runner传入的available_cameras和enable_lidar。
这是模型输入契约，不是缺失模态的零值替代。控制器据此创建真实相机和LiDAR，
实际同步仍由SynchronizedMultiviewCameraRig负责，未获得同步数据仍沿用传感器等待逻辑。
未声明该字段的配置保持runner原参数，便于旧实验复现。

最新challenge_signal_generalization配置已按模型README声明四视角和LiDAR。
决策日志sensor_contract保留启用视角、LiDAR开关及来源。雷达仍由既有观测层管理，
此字段不重新创建雷达，也不改变相机安装位置、分辨率或输入预处理。

若进行仅前视或无LiDAR消融，必须另存实验配置并明确其输入差异；不能称为完整模型配置复现。
配置通过不代表权重已齐备、传感器运行正常或模型性能达标。
