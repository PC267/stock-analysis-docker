# 上传到 GitHub

本目录已准备为 Git 仓库，可上传到 GitHub。

## 已包含
- `Dockerfile`：Web 选股镜像入口。
- `docker-compose.yml`：Web 服务编排配置。
- `.gitignore` / `.dockerignore`：排除运行时数据、报告、日志、venv、缓存。
- `.gitkeep`：保留必要空目录结构。

## 常用命令

```bash
cd /vol1/1000/docker/stock-analysis-docker
git status
git push origin main
```

## 注意
不要提交运行时数据、报告、日志、venv、IDE 配置和缓存文件。
