## 2026-08-18 - Task: 将教程笔记翻译为中文

### What was done

将仓库内的教程 Markdown 文档和有效 Jupyter Notebook 的 Markdown 单元翻译为简体中文；代码单元、执行输出、命令、公式、链接地址和 Notebook 元数据保持不变。

### Testing

- 73 个教程 Markdown 文件通过代码块和链接数量一致性检查。
- 66 个 Notebook 均通过 JSON 解析检查。
- 1,548 个 Markdown 单元中 1,356 个完成文字翻译。
- 逐单元比对确认非 Markdown 单元未发生变化。
- 检查未发现翻译占位符或“法学硕士”错误术语。

### Notes

- `README.md`、`troubleshooting.md`、`setup/`：翻译项目总览、故障排除和环境设置说明。
- `ch01/` 至 `ch07/`：翻译各章节及补充材料的 README、阅读说明和 Notebook Markdown 单元。
- `appendix-A/`、`appendix-B/`、`appendix-C/`、`appendix-D/`、`appendix-E/`：翻译附录说明和 Notebook Markdown 单元。
- `pkg/llms_from_scratch/README.md`：翻译可选 Python 包使用说明。
- 回滚方式：在确认当前工作区没有其他需要保留的改动后，执行 `git restore -- README.md troubleshooting.md setup ch01 ch02 ch03 ch04 ch05 ch06 ch07 appendix-A appendix-B appendix-C appendix-D appendix-E pkg/llms_from_scratch`；或仅恢复指定文件。


## 2026-10-14 - Task: 修复中文翻译的 Markdown 排版

### What was done

- 修复根 README 的 `<br>` 与提示引用块粘连，将完整中文书名号纳入书名链接文本。
- 恢复被合并的 HTML/正文、标题/正文、列表/正文、表头/正文及代码围栏边界。
- 恢复 GitHub 提示标记 `[!NOTE]`，修复被翻译的有序列表标点和缺失空格。
- 修复性能表格中重复的数值单元格及末尾分隔符，数值以 `main` 原文为准；统一另一处链接两侧括号的范围。

### Testing

- 对 PR 涉及的全部 73 个教程 Markdown 文件（另含本进度记录）进行检查，使用 Marked 的 GFM 解析器比对原文与译文的标题、引用块、列表/列表项及表格数量，全部一致。
- 比对 525 个解析后的 Markdown 链接/图片地址（包括重复项）及 HTML 标签/属性，全部与 `main` 一致；未改动链接地址。
- 表格数量、列数和行数与 `main` 一致。
- 193 个代码块的语言标记及内容与修复前翻译分支一致（比对前仅归一化原有粘连的围栏边界）；行内美元符号公式与修复前一致。
- 根 README 的 HTML 渲染结果确认提示内容位于 `<blockquote>` 内、目录生成 `<table>`，书名链接包含两侧书名号。
- `git diff --check` 通过；此次修改只涉及 Markdown，没有修改 Notebook 或可执行代码文件。

### Notes

- 这是排版与结构检查，未重新审校全部翻译措辞，也未检查外部链接的在线可达性。
- 比对基准：`main` 的 `ec0bac5e1d854306a8ea6c308da49aae22479bd1`；修复前翻译分支的 `52a97862bcae5af3ae2ef341a03e0bcad7d53d14`。
