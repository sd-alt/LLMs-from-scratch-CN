# Docker 环境搭建指南

如果您更喜欢隔离项目依赖项和配置的开发设置，那么使用 Docker 是一个非常有效的解决方案。这种方法消除了手动安装软件包和库的需要，并确保了一致的开发环境。

如果您更喜欢使用 [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) 和 [../02_installing-python-libraries](../02_installing-python-libraries) 中解释的 conda 方法，本指南将引导您完成为本书设置可选 docker 环境的过程。

<br>

## 下载并安装 Docker

开始使用 Docker 的最简单方法是为您的相关平台安装 [Docker Desktop](https://docs.docker.com/desktop/)。

Linux (Ubuntu) 用户可能更喜欢安装 [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)，并按照 [安装后](https://docs.docker.com/engine/install/linux-postinstall/) 步骤进行操作。

<br>

## 在 Visual Studio Code 中使用 Docker DevContainer

Docker DevContainer（即开发容器）是一种允许开发人员使用 Docker 容器作为成熟的开发环境的工具。这种方法确保用户可以快速启动并运行一致的开发环境，无论其本地计算机设置如何。

虽然 DevContainers 还可以与其他 IDE 配合使用，但与 DevContainers 配合使用的常用 IDE/编辑器是 Visual Studio Code (VS Code)。下面的指南解释了如何在 VS Code 上下文中使用本书的 DevContainer，但类似的过程也应适用于 PyCharm。 [如果您没有但想要使用它，请安装](https://code.visualstudio.com/download)。

1. 将此 GitHub 存储库和 `cd` 克隆到项目根目录中。

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
```

2. 将 `.devcontainer` 文件夹从 `setup/03_optional-docker-environment/` 移动到当前目录（项目根目录）。

```bash
mv setup/03_optional-docker-environment/.devcontainer ./
```

3. 在 Docker Desktop 中，确保 **_desktop-linux_ builder** 正在运行并将用于构建 Docker 容器（请参阅_Docker Desktop_ -> _更改设置_ -> _Builders_ -> _desktop-linux_ -> _..._ -> _Use_）

4. 如果您有 [支持 CUDA 的 GPU](https://developer.nvidia.com/cuda-gpus)，则可以加快训练和推理速度：

    4.1 按照此处的 [](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt) 的说明安装 **NVIDIA Container Toolkit**。支持 NVIDIA Container Toolkit，如此处所写的 [](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#nvidia-compute-software-support-on-wsl-2)。4.2 在 Docker 引擎守护程序配置中添加 _nvidia_ 作为运行时（请参阅_Docker 桌面_ -> _更改设置_ -> _Docker 引擎_）。将这些行添加到您的配置中：

    ```json
    "runtimes": {
        "nvidia": {
        "path": "nvidia-container-runtime",
        "runtimeArgs": []
    ```

    例如，完整的 Docker Engine 守护进程配置 json 代码应如下所示：

    ```json
    {
      "builder": {
        "gc": {
          "defaultKeepStorage": "20GB",
          "enabled": true
        }
      },
      "experimental": false,
      "runtimes": {
        "nvidia": {
          "path": "nvidia-container-runtime",
          "runtimeArgs": []
        }
      }
    }
    ```

    并重新启动 Docker Desktop。

5. 在终端中输入 `code .` 以在 VS Code 中打开项目。或者，您可以启动 VS Code 并从 UI 中选择要打开的项目。

6. 从左侧的 VS Code _Extensions_ 菜单安装 **Remote Development** 扩展。

7. 打开 DevContainer。

由于 `.devcontainer` 文件夹存在于主 `LLMs-from-scratch` 目录中（以 `.` 开头的文件夹可能在操作系统中不可见，具体取决于您的设置），VS Code 应该自动检测它并询问您是否要在 devcontainer 中打开项目。如果没有，只需按 `Ctrl + Shift + P` 打开命令面板并开始输入 `dev containers` 即可查看所有 DevContainer 特定选项的列表。


 
> ⚠️ **关于以 root 身份运行的注意事项**
>
> 默认情况下，DevContainer 以 *root 用户* 身份运行。出于安全原因，通常不建议这样做，但为了本书设置的简单性，使用根配置，以便所有必需的包都干净地安装在容器内。
>
> 如果您尝试在容器内手动启动 Jupyter Lab，您可能会看到以下错误：
>
> ```bash
>   Running as root is not recommended. Use --allow-root to bypass.
>   ```
>
> 在这种情况下，您可以运行：
>
> ```bash
>   uv run jupyter lab --allow-root
>   ```
>
> - 当使用 VS Code 和 Jupyter 扩展时，通常不需要手动启动 Jupyter Lab。通过扩展打开笔记本应该可以开箱即用。
> - 喜欢更严格安全性的高级用户可以修改 `.devcontainer.json` 以设置非 root 用户，但这需要额外的配置，并且对于大多数用例来说不是必需的。



8. 选择**在容器中重新打开**。

Docker 现在将开始构建 `.devcontainer` 配置中指定的 Docker 映像（如果之前尚未构建），或者拉取该映像（如果可从注册表获取）。

整个过程是自动化的，可能需要几分钟，具体取决于您的系统和互联网速度。可以选择单击 VS Code 右下角的“启动 Dev 容器（显示日志）”来查看当前构建进度。完成后，VS Code 将自动连接到容器并在新创建的 Docker 开发环境中重新打开项目。您将能够编写、执行和调试代码，就像在本地计算机上运行一样，但还具有 Docker 的隔离性和一致性的额外优势。

 
> **警告：**
> 如果您在构建过程中遇到错误，这可能是因为您的计算机不支持 NVIDIA 容器工具包，因为您的计算机没有兼容的 GPU。在这种情况下，编辑 `devcontainer.json` 文件以删除 `"runArgs": ["--runtime=nvidia", "--gpus=all"],` 行并再次运行“重新打开开发容器”过程。

9. 完成。

提取并构建映像后，您应该将项目安装在容器内，并安装所有软件包，准备进行开发。

<br>

## 卸载 Docker 镜像

如果您不再打算使用 Docker 容器和映像，以下是卸载或删除它的说明。此过程不会从系统中删除 Docker 本身，而是清理特定于项目的 Docker 工件。

1. 列出所有 Docker 镜像以查找与您的 DevContainer 关联的镜像：

```bash
docker image ls
```

2. 使用镜像 ID 或名称删除 Docker 镜像：

```bash
docker image rm [IMAGE_ID_OR_NAME]
```

<br>

## 卸载 Docker

如果您认为 Docker 不适合您并希望卸载它，请参阅此处的官方文档 [](https://docs.docker.com/desktop/uninstall/)，其中概述了适用于您的特定操作系统的步骤。