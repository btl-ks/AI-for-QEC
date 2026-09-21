# Verification

## Scope

本验证只证明仓库、OpenSpec、公共类型和 Protocol scaffold 一致且可导入。它不证明 syndrome 生成、数据集解析、训练、解码、恢复、统计评估或 Accuracy Gate 已实现。

## Evidence

### Contract tests

命令：

```bash
python3 -m unittest discover -s tests -v
```

结果：6 个测试全部通过，覆盖公共 facade 导入、frozen contract、Dataset 三层类型、checkpoint 类型分离、LER 证据字段以及不存在虚假执行入口。

### Import and syntax verification

命令：

```bash
python3 -m compileall -q ai_qec tests
```

结果：通过，无语法或导入错误。

### Capability registry schema

使用 `yaml.safe_load`、`json.loads` 与 `jsonschema.validate` 对 `openspec/capabilities.yaml` 和 `openspec/capabilities.schema.json` 进行验证。

结果：通过，共 6 个 capability；5 个领域 capability 均为 `planned/contracts-only`，治理 capability 为 `available/complete`。

### OpenSpec strict validation

命令：

```bash
openspec validate --all --strict --no-interactive
```

结果：`bootstrap-contract-repository` 通过，0 failed。

### Repository initialization

命令：

```bash
git init --initial-branch=main
git status --short --branch
```

结果：空 Git 仓库已在 `main` 分支初始化；工作树中的未跟踪内容均为本次 scaffold 或原有三份架构文档。未创建 commit。

## Capability Decision

- `governance/spec-management`：OpenSpec 工作流已经可用；完成归档后可以提升为 `validated`。
- 其他 capability：保持 `planned/contracts-only`，后续必须以独立 change 交付实现和科学证据。
