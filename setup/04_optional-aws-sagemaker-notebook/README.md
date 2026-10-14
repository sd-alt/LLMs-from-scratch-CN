# AWS CloudFormation 模板：带有 LLMs-from-scratch 存储库的 Jupyter Notebook

此 CloudFormation 模板在 Amazon SageMaker 中创建一个支持 GPU 的 Jupyter 笔记本，并具有执行角色和 LLMs-from-scratch GitHub 存储库。

## 它的作用：

1. 创建具有 SageMaker 笔记本实例所需权限的 IAM 角色。
2. 创建用于加密笔记本实例的 KMS 密钥和别名。
3. 配置笔记本实例生命周期配置脚本：
   - 在用户的主目录中安装单独的 Miniconda 安装。
   - 使用 TensorFlow 2.15.0 和 PyTorch 2.1.0 创建自定义 Python 环境，两者均支持 CUDA。
   - 安装其他软件包，如 Jupyter Lab、Matplotlib 和其他有用的库。
   - 将自定义环境注册为 Jupyter 内核。
4. 创建具有指定配置的 SageMaker 笔记本实例，包括启用 GPU 的实例类型、执行角色和默认代码存储库。

## 使用方法：

1. 下载CloudFormation模板文件（`cloudformation-template.yml`）。
2. 在 AWS 管理控制台中，导航到 CloudFormation 服务。
3. 创建新堆栈并上传模板文件。
4. 提供笔记本实例的名称（例如“LLMsFromScratchNotebook”）（默认为 LLMs-from-scratch GitHub 存储库）。
5. 检查并接受模板的参数，然后创建堆栈。
6. 堆栈创建完成后，SageMaker 笔记本实例将在 SageMaker 控制台中可用。
7. 打开笔记本实例并开始使用预配置的环境来处理从头开始的 LLM 项目。

## 要点：

- 该模板创建一个具有 50GB 存储空间的启用 GPU 的 (`ml.g4dn.xlarge`) 笔记本实例。
- 它使用 TensorFlow 2.15.0 和 PyTorch 2.1.0 设置自定义 Miniconda 环境，两者均支持 CUDA。
- 自定义环境注册为 Jupyter 内核，使其可在笔记本中使用。
- 该模板还创建用于加密笔记本实例的 KMS 密钥以及具有必要权限的 IAM 角色。