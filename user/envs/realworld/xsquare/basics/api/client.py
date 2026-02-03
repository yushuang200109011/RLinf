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

# 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobotControllerClient:
    """ - """
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        
        
        Args:
            server_url: 
        """
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        
        # 
        self.available_methods = self._get_available_methods()
        
        # 
        self._create_dynamic_methods()
    
    def _get_available_methods(self) -> Dict[str, Any]:
        """"""
        try:
            response = self.session.get(f"{self.server_url}/api/methods")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f": {e}")
            return {}
    
    def _create_dynamic_methods(self):
        """"""
        for method_name, method_info in self.available_methods.items():
            # 
            self._create_method(method_name, method_info)
    
    def _create_method(self, method_name: str, method_info: Dict[str, Any]):
        """"""
        signature = method_info.get('signature', {})
        
        def dynamic_method(*args, **kwargs):
            """"""
            # 
            if args:
                param_names = list(signature.keys())
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        kwargs[param_names[i]] = arg
            
            return self._call_remote_method(method_name, kwargs)
        
        # 
        dynamic_method.__doc__ = method_info.get('doc', f': {method_name}')
        dynamic_method.__name__ = method_name
        
        # 
        setattr(self, method_name, dynamic_method)
    
    def _call_remote_method(self, method_name: str, params: Dict[str, Any]) -> Any:
        """"""
        try:
            # 
            if not params:
                # GET
                response = self.session.get(f"{self.server_url}/api/{method_name}")
            else:
                # POST
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
                raise Exception(f": {result}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f" {method_name} : {e}")
            raise Exception(f": {e}")
    
    def get_available_methods(self) -> Dict[str, Any]:
        """"""
        return self.available_methods
    
    def health_check(self) -> Dict[str, Any]:
        """"""
        try:
            response = self.session.get(f"{self.server_url}/api/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f": {e}")
            return {"status": "unhealthy", "error": str(e)}

# 
if __name__ == "__main__":
    # 
    client = RobotControllerClient("http://localhost:8000")
    
    # 
    print(":", client.health_check())
    
    # 
    print("\n:")
    for method_name, method_info in client.get_available_methods().items():
        print(f"  {method_name}: {method_info['doc']}")
    
    # （）
    try:
        print("=====  =====")

        # 1. 
        print("\n1. :")
        print(" pitch=0.5, yaw=0.0")
        head_result = client.head_control([0.1, 0.0])
        print(":", head_result)
        
        # 2. 
        print("\n2. :")
        print("0.4")
        lift_result = client.lift_control(0.4)
        print(":", lift_result)
        
        # 3. 
        print("\n3. :")
        # 3.1 
        print(" [0.1,0.1,0.1,0.0,0.0,0.0,1.0],  [0.1,-0.1,0.1,0.0,0.0,0.0,1.0]")
        client.arms_control(
            [0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 1.0],
            [0.01, -0.01, 0.01, 0.0, 0.0, 0.0, 1.0]
        )
        
        # 3.2 
        print("\n:")
        print("")
        client.arms_zero()
        print("")
        
        # 4. 
        print("\n4. :")
        print("")
        client.chassis_set_current_pose_as_virtual_zero()
        print("")
        
        # 5. 
        print("\n5. :")
        # 5.1 
        head_data = client.head_data()
        print(f" - pitch: {head_data[0]}, yaw: {head_data[1]}")
        
        # 5.2 
        lift_data = client.lift_data()
        print(f": {lift_data}")
        
        # 5.3 
        arm_l, arm_r = client.arms_data()
        print(f": {arm_l}")
        print(f": {arm_r}")
        
        # 5.4 
        rel_pose = client.chassis_rel_pose_data()
        print(f": x={rel_pose[0]}, y={rel_pose[1]}, yaw={rel_pose[2]}")

        print("\n=====  =====")

    except Exception as e:
        stack_trace = traceback.format_exc()
        print(f"Error in arms_control: {str(e)}")
        print(stack_trace)
        print(f"\n: {str(e)}")

# #  - 
# class Turtle2ControllerClient(RobotControllerClient):
#     """Turtle2Controller"""
    
#     def __init__(self, server_url: str = "http://localhost:8000"):
#         super().__init__(server_url)
    
#     # 
#     def is_connected(self) -> bool:
#         """"""
#         health = self.health_check()
#         return health.get('status') == 'healthy'
    
#     def get_all_data(self) -> Dict[str, Any]:
#         """"""
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
#             logger.warning(f": {e}")
#         return data
    
#     def emergency_stop(self):
#         """（）"""
#         try:
#             # 
#             self.chassis_control_vel([0, 0, 0])
#             self.arms_zero()
#             logger.info("")
#         except Exception as e:
#             logger.error(f": {e}")

# # 
# if __name__ == "__main__":
#     # Turtle2
#     turtle_client = Turtle2ControllerClient("http://localhost:8000")
    
#     if turtle_client.is_connected():
#         print("Turtle2")
        
#         # 
#         all_data = turtle_client.get_all_data()
#         print(":", all_data)
        
#     else:
#         print("")