# 封装软件端的算法调用接口

## 使用示例
```
from weld_client_sdk import WeldClient

client = WeldClient("http://127.0.0.1:8000")
resp = client.pipeline(rgb_path, depth_path)
```

SDK 以 HTTP 方式调用 `weld_vision_server`，支持：
- `/category_recognition`
- `/dimension_measurement`
- `/pose_estimation`
- `/pipeline`
