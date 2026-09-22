# Design

## Context

见 `proposal.md`。当前正式 packaging spec 和 setuptools 配置使用 `src/ai_qec`，源码与测试树均为无 `__init__.py` 的 PEP 420 namespace package。公共 facade、叶子模块导入和隔离 wheel 验证已经稳定，本次只改变源码物理根目录和 discovery 边界。

## Goals / Non-Goals

**Goals:**

- 将源码恢复到根目录 `ai_qec`，允许从仓库根目录直接导入。
- 继续保持 setuptools、PEP 420 namespace package 和零 `__init__.py`。
- 确保根目录 package discovery 只收录 `ai_qec*`，不把测试、文档或 OpenSpec 目录打入 wheel。
- 保持公共 facade 的 130 个导出以及现有领域测试行为不变。

**Non-Goals:**

- 不恢复由 `__init__.py` 提供的包级聚合导入或版本属性。
- 不改变测试目录的 namespace 状态；unit 与 integration 仍分别发现。
- 不改变领域模块结构、Registry、runtime 依赖或科学能力状态。

## Decisions

### 使用根目录 discovery 并设置正向 include

删除 `tool.setuptools.package-dir` 的 `src` 映射；`tool.setuptools.packages.find` 使用 `where = ["."]`、`include = ["ai_qec*"]` 和 `namespaces = true`。正向 include 比依赖一组不断增长的 exclude 更安全，可直接限制 wheel 的 Python package namespace。

### 保留叶子模块导入和零初始化文件

源码移动不撤销上一变更已经完成的显式叶子模块导入。`ai_qec` 和 `tests` 中继续不创建 `__init__.py`；因此 `unittest` 仍分别以 `tests/unit` 与 `tests/integration` 为 start directory，避免顶层递归发现依赖普通 package 初始化文件。

### 同时验证直接源码导入与隔离 wheel 导入

根目录直接导入是此次新增的开发行为；测试将断言 facade 解析到根目录 `ai_qec/notebook_api.py`。发布边界仍通过仓库外临时虚拟环境安装 wheel 来验证，并检查 wheel 只有 `ai_qec/**.py` 与 dist-info Python 元数据、没有 `__init__.py` 或误收录的测试/文档 package。

### 保留向上查找项目标记的根目录解析

`find_project_root` 当前通过向上查找 `openspec/config.yaml` 解析源码检出，不再依赖固定父目录深度。该实现同时兼容根目录和历史 `src` layout，无需回退到脆弱的 `parents[n]`。

## Risks / Trade-offs

- [仓库根目录源码可能遮蔽已安装 distribution] → 文档明确该行为；发布验证必须从仓库外安装 wheel，不以根目录测试代替。
- [namespace discovery 可能收录无初始化文件的其他目录] → 使用 `include = ["ai_qec*"]` 正向限制，并检查 wheel 成员名称。
- [物理路径再次变化导致 evidence 或文档悬空] → 更新当前 capability evidence、README 与架构背景；已归档历史证据保持不变。
- [普通 `unittest discover -s tests` 不递归 namespace 测试目录] → 保留 unit/integration 分别执行的正式命令。

## Migration Plan

1. 修改 setuptools discovery 配置，将源码树机械移动回 `ai_qec`，并清理生成的空 `src` 构建目录。
2. 更新 packaging contract 测试与所有当前物理路径引用，确认项目内仍为零 `__init__.py`。
3. 从仓库根目录直接运行 compile、unit 和 integration 测试。
4. 重建 wheel，在仓库外隔离环境安装并验证 namespace、公共 facade、可选依赖惰性加载和 wheel 成员范围。
5. 写入 verification、更新 capability evidence、严格验证并归档；任一 gate 失败则恢复 `src` 配置与源码位置。
