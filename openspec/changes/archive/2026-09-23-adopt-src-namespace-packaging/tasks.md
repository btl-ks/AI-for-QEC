# Tasks

## 1. 构建与源码布局

- [x] 1.1 将构建后端配置为 setuptools 的 `src` layout 与 namespace discovery，并用 `uv build --wheel --no-build-isolation` 验证构建元数据可执行
- [x] 1.2 将 `ai_qec` 源码迁移到 `src/ai_qec`、删除源码与测试中的全部 `__init__.py`，并用 `test ! -d ai_qec` 和 `find src/ai_qec tests -name __init__.py` 验证唯一源码树与零初始化文件

## 2. 导入兼容与回归测试

- [x] 2.1 将 facade 和实现层的包级聚合导入改为叶子模块导入，并用 `python -m compileall -q src/ai_qec tests` 验证模块可编译
- [x] 2.2 增加 packaging contract 测试，覆盖 src 布局、零 `__init__.py`、namespace 语义和公共导出；用 `python -m unittest discover -s tests/unit -p test_packaging.py -v` 验证
- [x] 2.3 分别运行 `python -m unittest discover -s tests/unit -v` 与 `python -m unittest discover -s tests/integration -v`，验证领域行为、可选依赖惰性导入和公开 Notebook API 未回归
- [x] 2.4 构建 wheel，在仓库外临时环境安装并检查 wheel 内容、深层模块导入、公共 facade 与可选 runtime 未加载；保存命令输出作为 verification evidence

## 3. 文档、证据与治理

- [x] 3.1 更新 README、当前 capability evidence 路径和锁文件，并用 `rg '(^|[` ])ai_qec/' README.md openspec/capabilities.yaml pyproject.toml` 审核所有当前物理源码路径
- [x] 3.2 新增 `governance/python-packaging` capability 的实现状态与真实证据，并用 capability schema 测试及 `openspec validate --all --strict --no-interactive` 验证
- [x] 3.3 写入包含实际命令、结果与 wheel 隔离导入证据的 `verification.md`，完成任务清单后用 `openspec validate adopt-src-namespace-packaging --strict --no-interactive` 验证 change
