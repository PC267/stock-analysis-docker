# 股票智能模拟分析系统 - Docker部署

## 📋 系统概述

这是一个基于Python的股票智能模拟分析系统，使用baostock API获取真实股票数据，集成多种AI模型进行股票预测和模拟交易。

## 🚀 快速开始

### 1. 环境要求

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **磁盘空间**: 至少20GB（股票数据较大）
- **内存**: 至少4GB RAM
- **CPU**: 至少2核

### 2. 一键部署

```bash
# 克隆或下载项目
git clone https://github.com/PC267/stock-analysis-docker.git
cd stock-analysis-docker

# 快速启动（推荐）
make quick-start

# 或分步执行：
make init      # 初始化环境
make build     # 构建镜像
make up        # 启动服务
make health    # 检查状态
```

### 3. 配置说明

首次运行前需要配置 `.env` 文件：

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置文件
vim .env
```

**重要配置项**：
- `BAOSTOCK_USER` / `BAOSTOCK_PASSWORD`: baostock账号（需要注册）
- `DB_PASSWORD`: 数据库密码
- `REDIS_PASSWORD`: Redis密码
- `STOCK_CODE`: 默认分析的股票代码

### 4. 访问服务

| 服务 | 地址 | 默认端口 | 说明 |
|------|------|----------|------|
| 主应用 | localhost:8822 | 8822 | 股票分析系统 |
| PostgreSQL | localhost:5432 | 5432 | 数据库 |
| Redis | localhost:6379 | 6379 | 缓存服务 |

## 🏗️ 系统架构

### 服务组件

```
┌─────────────────────────────────────────────────────────────┐
│                   股票智能模拟分析系统架构                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐       │
│  │  主应用     │◄───┤ PostgreSQL │◄───┤   Redis    │       │
│  │ (分析引擎)  │    │  (数据库)   │    │  (缓存)    │       │
│  └────────────┘    └────────────┘    └────────────┘       │
│         │                                                    │
│         ▼                                                    │
│  ┌────────────┐                                             │
│  │ baostock   │                                             │
│  │ (数据源)   │                                             │
│  └────────────┘                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

1. **数据获取** → baostock API → 本地缓存
2. **特征工程** → 技术指标计算 → 特征存储
3. **模型预测** → 多种AI模型 → 结果分析
4. **回测验证** → 历史数据回测 → 性能评估
5. **报告生成** → 可视化图表 → 结果报告

## 📊 主要功能

### 1. 数据获取与处理
- 从baostock获取股票历史数据
- 数据清洗和预处理
- 技术指标计算（RSI, MACD, 布林带等）

### 2. AI模型预测
- **时间序列模型**: ARIMA, SARIMA
- **机器学习模型**: XGBoost, Random Forest
- **集成学习**: 动态权重集成
- **概率预测**: 不确定性量化

### 3. 回测与模拟
- 历史数据回测
- 模拟交易策略
- 风险收益分析
- 性能指标计算

### 4. 报告生成
- 可视化图表
- 分析报告
- 投资建议
- 风险提示

## 🔧 技术栈

### 核心依赖
- **Python 3.11**
- **pandas / numpy**: 数据处理
- **scikit-learn / xgboost**: 机器学习
- **statsmodels**: 时间序列分析
- **matplotlib**: 数据可视化

### 数据源
- **baostock**: 免费股票数据API
- **本地缓存**: 减少API调用

### 基础设施
- **Docker**: 容器化部署
- **PostgreSQL**: 数据存储
- **Redis**: 缓存服务

## 📁 目录结构

```
stock-analysis-docker/
├── docker/                    # Docker配置
│   └── Dockerfile            # Python应用Dockerfile
├── src/                      # 源代码
│   ├── app/                  # 应用代码
│   │   ├── main.py          # 主程序入口
│   │   ├── stock_analysis.py # 股票分析核心
│   │   ├── backtest.py      # 回测模块
│   │   ├── data/            # 数据处理
│   │   ├── model/           # AI模型
│   │   ├── features/        # 特征工程
│   │   ├── ensemble/        # 集成学习
│   │   └── report_chart/    # 报告生成
│   └── config/              # 配置文件
│       └── config.yaml      # 系统配置
├── data/                     # 数据目录
│   ├── stock_data/          # 股票数据缓存
│   ├── results/             # 分析结果
│   └── models/              # 训练好的模型
├── logs/                     # 日志目录
├── .env.example             # 环境变量示例
├── requirements.txt         # Python依赖
├── docker-compose.yml       # 服务编排
├── Makefile                 # 自动化脚本
└── README.md               # 本文档
```

## 🛠️ 使用指南

### 1. 运行股票分析

```bash
# 运行完整的股票分析
make run-analysis

# 或手动执行
docker-compose exec stock-analysis python src/app/main.py
```

### 2. 运行回测

```bash
# 运行回测分析
make run-backtest

# 或手动执行
docker-compose exec stock-analysis python src/app/backtest.py
```

### 3. 数据管理

```bash
# 导入股票数据
make import-data

# 备份数据
make backup

# 恢复数据
make restore BACKUP_FILE=backups/db_20240101_120000.sql

# 清除缓存
make clear-cache
```

### 4. 系统管理

```bash
# 查看服务状态
make ps
make health
make stats

# 查看日志
make logs
make logs-app
make logs-db

# 进入容器
make shell      # 进入应用容器
make shell-db   # 进入数据库
```

## ⚙️ 配置说明

### 配置文件

系统有两个主要配置文件：

1. **`.env`**: 环境变量配置
   - 数据库连接
   - API密钥
   - 性能参数

2. **`src/config/config.yaml`**: 系统配置
   - 股票代码
   - 分析参数
   - 模型配置

### 修改股票代码

编辑 `src/config/config.yaml`：

```yaml
stock_system:
  stock:
    code: '000001'    # 修改为你要分析的股票代码
    name: 平安银行     # 股票名称
```

## 📈 输出结果

分析结果保存在 `data/results/` 目录：

```
data/results/
├── charts/              # 可视化图表
├── reports/             # 分析报告
├── backtest_results/    # 回测结果
└── predictions/         # 预测结果
```

## 🚨 故障排除

### 常见问题

**问题1: baostock连接失败**
```bash
# 检查baostock账号配置
echo $BAOSTOCK_USER
echo $BAOSTOCK_PASSWORD

# 测试baostock连接
docker-compose exec stock-analysis python -c "import baostock as bs; bs.login()"
```

**问题2: 内存不足**
```bash
# 查看内存使用
make stats

# 调整内存限制
# 修改.env中的MEMORY_LIMIT
MEMORY_LIMIT=8G
```

**问题3: 数据下载缓慢**
```bash
# baostock API有频率限制
# 可以调整下载间隔
# 修改config.yaml中的下载参数
```

**问题4: 依赖安装失败**
```bash
# 重新构建镜像
make build-no-cache

# 或手动安装依赖
docker-compose exec stock-analysis pip install -r requirements.txt
```

### 日志分析

```bash
# 查看错误日志
grep -i error logs/app.log

# 查看数据下载日志
grep "baostock" logs/app.log

# 查看分析进度
tail -f logs/app.log | grep "进度"
```

## 🔄 更新与维护

### 系统更新

```bash
# 拉取最新代码
git pull origin main

# 重新构建
make build
make up

# 验证更新
make test
make health
```

### 定期维护

```bash
# 每日备份
make backup

# 清理旧数据
find ./data/results -type f -mtime +30 -delete

# 清理日志
find ./logs -type f -mtime +7 -delete
```

### 监控建议

1. **磁盘空间**: 股票数据可能占用大量空间
2. **API调用**: baostock有调用频率限制
3. **内存使用**: AI模型训练需要较多内存
4. **网络连接**: 需要稳定的网络连接获取数据

## 📚 相关资源

### 文档
- [baostock API文档](http://baostock.com/baostock/index.php/)
- [系统配置说明](./docs/CONFIGURATION.md)
- [API接口文档](./docs/API.md)

### 工具
- [Docker文档](https://docs.docker.com/)
- [Docker Compose文档](https://docs.docker.com/compose/)

### 支持
- 问题反馈: [GitHub Issues](https://github.com/your-repo/issues)
- 技术交流: [Discord社区](https://discord.gg/your-community)

## 🆘 技术支持

### 遇到问题？

1. 查看日志: `make logs`
2. 检查状态: `make health`
3. 验证配置: 检查 `.env` 和 `config.yaml`
4. 查阅文档: [FAQ](./docs/FAQ.md)

### 紧急联系方式

- **技术支持**: 181665281@qq.com

---

**最后更新**: 2026-04-09  
**版本**: 1.0.0  
**维护者**: 股票智能模拟分析系统团队
