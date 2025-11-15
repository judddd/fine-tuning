"""
Model Comparison Tool - FastAPI Backend
用于对比两个本地模型的推理效果
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import logging
import asyncio
from pathlib import Path

from model_manager import ModelManager, list_all_adapters

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="Model Comparison Tool",
    description="Compare two fine-tuned models side by side",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型管理器
model_managers: Dict[str, ModelManager] = {
    "model_a": None,
    "model_b": None
}


# ==================== Pydantic Models ====================

class ModelConfig(BaseModel):
    model_path: str
    model_name: Optional[str] = None
    adapter_path: Optional[str] = None  # 适配器文件路径
    no_adapter: bool = False  # 是否不使用适配器（使用基础模型）
    saves_dir: Optional[str] = None  # 适配器保存目录（用于自动查找，默认: mlx/saves/qwen-lora）


class GenerateRequest(BaseModel):
    prompt: str
    model_id: str  # "model_a" or "model_b"
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9


# ==================== API Endpoints ====================

@app.get("/")
async def read_root():
    """主页"""
    return FileResponse("static/index.html")


@app.post("/api/models/{model_id}/load")
async def load_model(model_id: str, config: ModelConfig):
    """
    加载模型到内存
    
    Args:
        model_id: "model_a" 或 "model_b"
        config: 模型配置（路径和名称）
    """
    if model_id not in ["model_a", "model_b"]:
        raise HTTPException(status_code=400, detail="Invalid model_id. Use 'model_a' or 'model_b'")
    
    # 卸载旧模型（如果存在）
    if model_managers[model_id] is not None:
        logger.info(f"Unloading existing {model_id}...")
        model_managers[model_id].unload_model()
    
    # 创建新模型管理器
    model_name = config.model_name or f"Model {model_id.upper()}"
    saves_dir = config.saves_dir or "mlx/saves/qwen-lora"
    manager = ModelManager(
        model_path=config.model_path,
        model_name=model_name,
        adapter_path=config.adapter_path,
        no_adapter=config.no_adapter,
        saves_dir=saves_dir
    )
    
    # 异步加载模型（在后台线程执行以避免阻塞）
    logger.info(f"Loading {model_id} from {config.model_path}...")
    
    # 在 executor 中运行阻塞的加载操作
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, manager.load_model)
    
    if success:
        model_managers[model_id] = manager
        return {
            "success": True,
            "message": f"Model '{model_name}' loaded successfully",
            "model_info": manager.get_model_info()
        }
    else:
        return {
            "success": False,
            "message": f"Failed to load model from {config.model_path}",
            "error": "Check server logs for details"
        }


@app.post("/api/models/{model_id}/unload")
async def unload_model(model_id: str):
    """卸载模型释放内存"""
    if model_id not in ["model_a", "model_b"]:
        raise HTTPException(status_code=400, detail="Invalid model_id")
    
    if model_managers[model_id] is None:
        return {"success": False, "message": "Model not loaded"}
    
    model_managers[model_id].unload_model()
    model_managers[model_id] = None
    
    return {"success": True, "message": f"{model_id} unloaded"}


@app.get("/api/models/{model_id}/status")
async def get_model_status(model_id: str):
    """获取模型状态"""
    if model_id not in ["model_a", "model_b"]:
        raise HTTPException(status_code=400, detail="Invalid model_id")
    
    manager = model_managers[model_id]
    
    if manager is None:
        return {
            "loaded": False,
            "model_id": model_id
        }
    
    return {
        "loaded": manager.is_loaded(),
        "model_id": model_id,
        "model_info": manager.get_model_info()
    }


@app.get("/api/status")
async def get_all_status():
    """获取所有模型状态"""
    return {
        "model_a": {
            "loaded": model_managers["model_a"] is not None and model_managers["model_a"].is_loaded(),
            "info": model_managers["model_a"].get_model_info() if model_managers["model_a"] else None
        },
        "model_b": {
            "loaded": model_managers["model_b"] is not None and model_managers["model_b"].is_loaded(),
            "info": model_managers["model_b"].get_model_info() if model_managers["model_b"] else None
        }
    }


@app.get("/api/adapters")
async def get_adapters(saves_dir: Optional[str] = None):
    """
    获取所有可用的适配器列表
    
    Args:
        saves_dir: 适配器保存目录（可选，默认: mlx/saves/qwen-lora）
    
    Returns:
        适配器列表，按时间排序（最新的在前）
    """
    saves_dir = saves_dir or "mlx/saves/qwen-lora"
    adapters = list_all_adapters(saves_dir)
    return {
        "adapters": adapters,
        "count": len(adapters)
    }


@app.post("/api/generate/stream")
async def generate_stream(request: GenerateRequest):
    """
    流式生成文本
    
    使用 Server-Sent Events (SSE) 返回流式响应
    """
    if request.model_id not in ["model_a", "model_b"]:
        raise HTTPException(status_code=400, detail="Invalid model_id")
    
    manager = model_managers[request.model_id]
    
    if manager is None or not manager.is_loaded():
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model_id} is not loaded"
        )
    
    async def event_generator():
        """SSE 事件生成器"""
        try:
            # 在 executor 中运行生成器
            loop = asyncio.get_event_loop()
            
            for text_chunk in manager.generate_stream(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ):
                # 发送 SSE 格式数据
                yield f"data: {text_chunk}\n\n"
                await asyncio.sleep(0)  # 让出控制权
            
            # 发送结束标记
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# 挂载静态文件
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==================== 启动事件 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动"""
    logger.info("=" * 60)
    logger.info("🚀 Model Comparison Tool Starting...")
    logger.info("=" * 60)
    logger.info("📊 Dashboard: http://localhost:8100")
    logger.info("📚 API Docs: http://localhost:8100/docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭"""
    logger.info("Shutting down... Unloading models...")
    for model_id in ["model_a", "model_b"]:
        if model_managers[model_id] is not None:
            model_managers[model_id].unload_model()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,
        log_level="info"
    )

