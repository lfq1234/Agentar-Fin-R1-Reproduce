"""聚合服务层入口。

注意：本包 __init__ 仅承载文档说明，不在此主动 import ``chat_service`` /
``analyze_service`` 等业务子模块——它们会经 ``app.agent`` 拉起 agentscope 等重
依赖。业务子模块由各自的路由/调用方按需直接 ``from app.services.xxx import``，
避免任何 ``import app.services``（含其子包，如 ``app.db.history``）都被动
加载重依赖。
"""
