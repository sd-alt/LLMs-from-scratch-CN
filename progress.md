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
