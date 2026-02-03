from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import inspect
import asyncio
import json

import os
import traceback
import traceback
from turtle2_controller.Turtle2Controller import Turtle2Controller
import uvicorn
import logging
from datetime import datetime

# 导入您的控制器类
# from your_robot_module import Turtle2Controller

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobotControllerServer:
    """机器人控制器服务器 - 自动暴露控制器类的所有方法"""
    
    def __init__(self, controller_class, controller_init_args=None, controller_init_kwargs=None):
        """
        初始化服务器
        
        Args:
            controller_class: 控制器类 (如 Turtle2Controller)
            controller_init_args: 控制器初始化参数
            controller_init_kwargs: 控制器初始化关键字参数
        """
        self.controller_class = controller_class
        self.controller_init_args = controller_init_args or []
        self.controller_init_kwargs = controller_init_kwargs or {}
        
        # 创建FastAPI应用
        self.app = FastAPI(
            title="Robot Controller API",
            description="自动生成的机器人控制器API",
            version="1.0.0"
        )
        
        # 初始化控制器实例
        self.controller = None
        self._initialize_controller()
        
        # 自动发现并注册所有方法
        self._register_methods()
        
        # 添加基础路由
        self._add_base_routes()
    
    def _initialize_controller(self):
        """初始化控制器实例"""
        try:
            self.controller = self.controller_class(
                *self.controller_init_args, 
                **self.controller_init_kwargs
            )
            logger.info(f"控制器 {self.controller_class.__name__} 初始化成功")
        except Exception as e:
            logger.error(f"控制器初始化失败: {e}")
            raise
    
    def _get_method_signature(self, method):
        """获取方法签名信息"""
        sig = inspect.signature(method)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_info = {
                'name': param_name,
                'required': param.default == inspect.Parameter.empty,
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any'
            }
            params[param_name] = param_info
        
        return params
    
    def _register_methods(self):
        """自动发现并注册控制器的所有公共方法"""
        # 获取所有公共方法
        print(f"开始获取方法...")
        methods = inspect.getmembers(self.controller, predicate=inspect.ismethod)
        for method_name, method in methods:
            # 跳过私有方法和特殊方法
            if method_name.startswith('_'):
                continue
            
            # 获取方法签名
            method_signature = self._get_method_signature(method)
            
            # 创建动态路由
            self._create_dynamic_route(method_name, method, method_signature)
            
            print(f"注册方法: {method_name}")
    
    def _create_dynamic_route(self, method_name, method, method_signature):
        """为每个方法创建动态路由"""
        
        # 创建请求模型
        request_fields = {}
        for param_name, param_info in method_signature.items():
            if param_info['required']:
                request_fields[param_name] = (Any, ...)
            else:
                request_fields[param_name] = (Any, param_info['default'])
        
        # 动态创建Pydantic模型
        RequestModel = None
        if request_fields:
            RequestModel = type(f"{method_name.capitalize()}Request", (BaseModel,), {
                '__annotations__': {k: v[0] for k, v in request_fields.items()},
                **{k: v[1] for k, v in request_fields.items() if v[1] is not ...}
            })
        
        # 创建路由处理函数
        async def route_handler(request: RequestModel = None):
            try:
                # 准备参数
                if request:
                    kwargs = request.dict()
                    # 过滤掉None值（对于可选参数）
                    kwargs = {k: v for k, v in kwargs.items() if v is not None}
                else:
                    kwargs = {}
                
                # 调用实际方法
                if asyncio.iscoroutinefunction(method):
                    result = await method(**kwargs)
                else:
                    result = method(**kwargs)
                print(type(result))
                # 特殊处理二进制结果
                if isinstance(result, bytes):
                    from fastapi.responses import Response
                    return Response(
                        content=result,
                        media_type="application/octet-stream"  # 或 "image/jpeg"
                    )
                
                # 返回结果
                return {
                    "success": True,
                    "result": result,
                    "method": method_name,
                    "timestamp": datetime.now().isoformat()
                }
                

            except Exception as e:
                # 记录完整的错误堆栈
                logger.error(f"调用方法 {method_name} 时发生错误", exc_info=True)
                
                # 获取堆栈信息并确保它是可序列化的
                stack_trace = traceback.format_exc()
                print(stack_trace)
                
                # 确保所有错误详情都是基本类型
                error_detail = {
                    "error": str(e),
                    "type": type(e).__name__,
                    "method": method_name,
                    "timestamp": datetime.now().isoformat()
                }
                
                # 只在调试模式下返回堆栈跟踪
                if os.getenv("DEBUG", "").lower() in ("true", "1", "t"):
                    error_detail["stack_trace"] = stack_trace
                
                raise HTTPException(
                    status_code=500,
                    detail=error_detail
                )
        
        # 注册路由
        if RequestModel:
            self.app.post(f"/api/{method_name}")(route_handler)
        else:
            # 对于无参数方法，使用GET请求
            async def get_handler():
                return await route_handler()
            self.app.get(f"/api/{method_name}")(get_handler)
    
    def _add_base_routes(self):
        """添加基础路由"""
        
        @self.app.get("/")
        async def root():
            return {"message": "Robot Controller API Server", "status": "running"}
        
        @self.app.get("/api/methods")
        async def get_methods():
            """获取所有可用方法及其签名"""
            methods = inspect.getmembers(self.controller, predicate=inspect.ismethod)
            method_info = {}
            
            for method_name, method in methods:
                if method_name.startswith('_'):
                    continue
                
                method_signature = self._get_method_signature(method)
                method_info[method_name] = {
                    "signature": method_signature,
                    "doc": inspect.getdoc(method) or "无文档",
                    "endpoint": f"/api/{method_name}"
                }
            
            return method_info
        
        @self.app.get("/api/health")
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "controller": self.controller_class.__name__,
                "timestamp": datetime.now().isoformat()
            }

# 使用示例
if __name__ == "__main__":
    # 模拟控制器类（替换为您的实际导入）
    # 创建服务器实例
    # 使用您的实际控制器类替换 MockTurtle2Controller
    server = RobotControllerServer(
        controller_class=Turtle2Controller,
        controller_init_kwargs={'init_node': True}
    )
    
    # 启动服务器
    uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )