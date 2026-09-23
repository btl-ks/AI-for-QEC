# Tasks

## 1. Registry namespace 迁移

- [x] 1.1 将四个顶层 Registry 子系统模块迁移为 `ai_qec.registry` 的明确叶子模块，并用 `find ai_qec/registry -maxdepth 1 -type f` 验证文件布局且不存在 `__init__.py`
- [x] 1.2 更新全部源码导入与 README 路径，并用 `rg 'ai_qec\.(config_validation|implementations|registries)|from ai_qec\.registry import' ai_qec tests README.md` 验证不再引用旧内部路径

## 2. 契约测试

- [x] 2.1 更新 Registry 与打包测试以覆盖新叶子模块、旧文件移除和公共 facade 不变性，并用 `python -m unittest tests.unit.test_registry tests.unit.test_packaging` 验证
- [x] 2.2 运行全量测试与静态检查以审计仓库基线，并用 `python -m unittest` 对非 Notebook 单元测试、`ruff check ai_qec/registry` 与 `pyright ai_qec/registry` 验证本 change 相关范围通过；既有无关失败记录到 `verification.md`

## 3. 分发与 OpenSpec 验证

- [x] 3.1 构建 wheel 并在仓库外临时环境检查 Registry 叶子模块、公共导出与延迟加载行为，结果记录到 `verification.md`
- [x] 3.2 运行 `openspec validate reorganize-registry-package --strict --no-interactive` 和 `openspec validate --all --strict --no-interactive`，在 `verification.md` 记录命令及真实结果
