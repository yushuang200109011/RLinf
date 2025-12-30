# client.py
import json
import time
import traceback
import requests
import inspect
from typing import List, Any, Tuple, Dict
# from turtle2_controller.Turtle2Controller import Turtle2Controller

import requests
import json
from typing import Any, Dict, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobotControllerClient:
    """机器人控制器客户端 - 提供与服务端相同的接口"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        初始化客户端
        
        Args:
            server_url: 服务端地址
        """
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        
        # 获取服务端可用方法
        self.available_methods = self._get_available_methods()
        
        # 动态创建方法
        self._create_dynamic_methods()
    
    def _get_available_methods(self) -> Dict[str, Any]:
        """获取服务端可用方法"""
        try:
            response = self.session.get(f"{self.server_url}/api/methods")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"获取方法列表失败: {e}")
            return {}
    
    def _create_dynamic_methods(self):
        """根据服务端方法动态创建客户端方法"""
        for method_name, method_info in self.available_methods.items():
            # 创建动态方法
            self._create_method(method_name, method_info)
    
    def _create_method(self, method_name: str, method_info: Dict[str, Any]):
        """创建单个方法"""
        signature = method_info.get('signature', {})
        
        def dynamic_method(*args, **kwargs):
            """动态生成的方法"""
            # 处理位置参数
            if args:
                param_names = list(signature.keys())
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        kwargs[param_names[i]] = arg
            
            return self._call_remote_method(method_name, kwargs)
        
        # 设置方法文档
        dynamic_method.__doc__ = method_info.get('doc', f'调用远程方法: {method_name}')
        dynamic_method.__name__ = method_name
        
        # 添加到类实例
        setattr(self, method_name, dynamic_method)
    
    def _call_remote_method(self, method_name: str, params: Dict[str, Any]) -> Any:
        """调用远程方法"""
        try:
            # 判断是否为无参数方法
            if not params:
                # 无参数方法使用GET请求
                response = self.session.get(f"{self.server_url}/api/{method_name}")
            else:
                # 有参数方法使用POST请求
                response = self.session.post(
                    f"{self.server_url}/api/{method_name}",
                    json=params,
                    headers={'Content-Type': 'application/json'}
                )
            
            response.raise_for_status()
            result = response.json()
            
            if result.get('success', False):
                return result.get('result')
            else:
                raise Exception(f"远程方法调用失败: {result}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"调用远程方法 {method_name} 失败: {e}")
            raise Exception(f"网络请求失败: {e}")
    
    def get_available_methods(self) -> Dict[str, Any]:
        """获取可用方法列表"""
        return self.available_methods
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            response = self.session.get(f"{self.server_url}/api/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"健康检查失败: {e}")
            return {"status": "unhealthy", "error": str(e)}

# 使用示例
if __name__ == "__main__":
    # 创建客户端实例
    client = RobotControllerClient("http://localhost:8000")
    
    # 检查服务器健康状态
    print("健康检查:", client.health_check())
    
    # 查看可用方法
    print("\n可用方法:")
    for method_name, method_info in client.get_available_methods().items():
        print(f"  {method_name}: {method_info['doc']}")
    
    # 调用示例（这些方法会根据服务端自动生成）
    try:
        print("===== 测试所有状态获取功能 =====")

        # 1. 测试头部控制
        print("\n1. 测试头部控制:")
        print("设置头部位置 pitch=0.5, yaw=0.0")
        head_result = client.head_control([0.1, 0.0])
        print("当前头部位置:", head_result)
        
        # 2. 测试升降机控制
        print("\n2. 测试升降机控制:")
        print("设置升降机高度为0.4")
        lift_result = client.lift_control(0.4)
        print("当前升降机高度:", lift_result)
        
        # 3. 测试机械臂控制
        print("\n3. 测试机械臂控制:")
        # 3.1 直接控制机械臂
        print("设置左臂位置 [0.1,0.1,0.1,0.0,0.0,0.0,1.0], 右臂位置 [0.1,-0.1,0.1,0.0,0.0,0.0,1.0]")
        client.arms_control(
            [0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 1.0],
            [0.01, -0.01, 0.01, 0.0, 0.0, 0.0, 1.0]
        )
        
        # 3.2 测试机械臂归零
        print("\n测试机械臂归零:")
        print("双臂直接归零")
        client.arms_zero()
        print("归零完成")
        
        # 4. 测试虚拟零点设置
        print("\n4. 测试虚拟零点设置:")
        print("设置当前底盘位置为虚拟零点")
        client.chassis_set_current_pose_as_virtual_zero()
        print("虚拟零点设置完成")
        
        # 5. 验证所有设置结果
        print("\n5. 验证所有设置结果:")
        # 5.1 头部状态
        head_data = client.head_data()
        print(f"头部位置 - pitch: {head_data[0]}, yaw: {head_data[1]}")
        
        # 5.2 升降机状态
        lift_data = client.lift_data()
        print(f"升降机高度: {lift_data}")
        
        # 5.3 机械臂状态
        arm_l, arm_r = client.arms_data()
        print(f"左臂末端: {arm_l}")
        print(f"右臂末端: {arm_r}")
        
        # 5.4 底盘相对位置
        rel_pose = client.chassis_rel_pose_data()
        print(f"底盘相对位置: x={rel_pose[0]}, y={rel_pose[1]}, yaw={rel_pose[2]}")

        print("\n===== 所有设置位置功能测试完成 =====")

    except Exception as e:
        stack_trace = traceback.format_exc()
        print(f"Error in arms_control: {str(e)}")
        print(stack_trace)
        print(f"\n操作失败: {str(e)}")

# # 高级客户端包装器 - 提供更像原始类的接口
# class Turtle2ControllerClient(RobotControllerClient):
#     """专门为Turtle2Controller设计的客户端包装器"""
    
#     def __init__(self, server_url: str = "http://localhost:8000"):
#         super().__init__(server_url)
    
#     # 可以添加一些客户端特定的辅助方法
#     def is_connected(self) -> bool:
#         """检查是否连接到服务器"""
#         health = self.health_check()
#         return health.get('status') == 'healthy'
    
#     def get_all_data(self) -> Dict[str, Any]:
#         """获取所有传感器数据"""
#         data = {}
#         try:
#             data['head'] = self.head_data()
#             data['arms'] = self.arms_data()
#             data['lift'] = self.lift_data()
#             data['chassis'] = self.chassis_pose_data()
#             data['cam1'] = self.cam1_data()
#             data['cam2'] = self.cam2_data()
#             data['cam3'] = self.cam3_data()
#         except Exception as e:
#             logger.warning(f"获取部分数据失败: {e}")
#         return data
    
#     def emergency_stop(self):
#         """紧急停止（如果服务端支持）"""
#         try:
#             # 停止所有运动
#             self.chassis_control_vel([0, 0, 0])
#             self.arms_zero()
#             logger.info("紧急停止执行完成")
#         except Exception as e:
#             logger.error(f"紧急停止失败: {e}")

# # 使用高级客户端的示例
# if __name__ == "__main__":
#     # 创建Turtle2专用客户端
#     turtle_client = Turtle2ControllerClient("http://localhost:8000")
    
#     if turtle_client.is_connected():
#         print("已连接到Turtle2控制器服务器")
        
#         # 获取所有数据
#         all_data = turtle_client.get_all_data()
#         print("所有传感器数据:", all_data)
        
#     else:
#         print("无法连接到服务器")