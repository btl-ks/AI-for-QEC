# Design

## Context

当前项目 `AGENTS.md` 已声明 OpenSpec 的线性交付路径，但没有定义自然语言动作动词的授权边界、需求到证据追踪细节或完整的归档阻断条件。Zephy 级 `/home/zephy/.codex/AGENTS.md` 目前只包含文本格式规则。旧项目中存在经过实际使用的相关规则，但其生成型 `.agents/skills` 属于工具状态，不能直接作为当前项目的正式规范复制。

## Goals / Non-Goals

**Goals:**

- 在共享层统一“记录、分析、设计、实现、继续”等意图语义。
- 在项目层固化完整 OpenSpec 交付门禁，并保持 QEC 专属规则优先和可见。
- 让“实现/完成”请求一次授权完整生命周期，同时保护纯设计请求不越界。
- 通过文本检查和严格 OpenSpec validation 验证两层规范。

**Non-Goals:**

- 不修改 QEC Python 实现、配置 contract、数据或运行产物。
- 不把 OpenSpec 强制推广到其他未采用 OpenSpec 的项目。
- 不直接复制旧项目生成的 `.agents/skills`，也不修改 OpenSpec CLI 全局安装。

## Decisions

### 1. 通用规则放在 Zephy 级，项目规则保留完整可执行版本

Zephy 级规范记录跨项目通用的用户意图边界，并以“项目明确采用 OpenSpec”为启用 OpenSpec 门禁的条件。项目 `AGENTS.md` 保留完整的 QEC OpenSpec 流程和归档条件，使单独读取项目规范时仍可执行。

替代方案是只在全局规范引用项目文件，但这会让其他 OpenSpec 项目无法复用通用语义，也会使项目规范脱离全局文件后不完整。

### 2. 自然语言授权与生成型工作流名称分离

规范以用户意图为授权源，不要求用户记忆 `apply`、`verify` 等工具名称。OpenSpec CLI 和生成型技能只是执行机制；纯 `propose` 请求保持规划边界，而明确的“实现/完成”请求在规划校验后自动继续。

这避免工具命令成为额外授权关卡，也保留用户只要求设计时的安全边界。

### 3. 规划校验和实现验证是独立门禁

`openspec validate --strict` 只证明规划材料有效；测试与 verify workflow 证明实现满足规划。`verification.md` 记录命令、结果、证据和限制。归档必须同时满足两类门禁。

### 4. 外部共享规范作为显式、可核验的用户配置修改

项目 change 记录共享规范同步的设计和验证，但 `/home/zephy/.codex/AGENTS.md` 位于仓库外，不进入 Git。实施时先生成完整目标内容，再通过受控权限写入；验证记录其路径、文本断言和摘要。回滚方式是恢复修改前的短文件内容。

## Risks / Trade-offs

- [全局规则影响其他项目] → 仅在项目已声明采用 OpenSpec 时启用 OpenSpec 门禁，并明确项目局部规则继续生效。
- [项目与全局内容未来漂移] → 正式要求关注语义一致而非逐字一致；项目层可增加 QEC 专属约束。
- [生成型技能与自然语言语义冲突] → 规范明确用户原始实现授权持续有效，纯提案请求仍只规划。
- [仓库外文件不能由 Git 回滚] → 在 verification 中记录修改后的摘要和检查命令，且保持修改范围仅限新增规范章节。

## Migration Plan

1. 扩充项目 `AGENTS.md` 的 OpenSpec Workflow、用户意图、追踪链、交付门禁和 archive blockers。
2. 在 Zephy 级 `AGENTS.md` 中加入通用用户意图与条件式 OpenSpec 工作流，不改变现有文本格式规则。
3. 运行文本断言、严格 change validation 与全仓 OpenSpec validation。
4. 生成 `verification.md`、同步主规格并归档 change。
