# SAR WAKE GitHub Pages

这是一个无构建依赖的中英文静态研究展示站，可直接部署到 GitHub Pages。

- `index.html`：默认英文主页
- `zh-CN.html`：完整中文页面
- 顶部导航中的 `中文 / EN` 按钮用于切换语言

## 最快部署方式

1. 将本目录中的所有文件（包括 `.github` 与 `.nojekyll`）上传到 GitHub 仓库根目录。
2. 打开仓库 **Settings → Pages**。
3. 在 **Build and deployment** 中将 Source 设为 **GitHub Actions**。
4. 推送后，随附的工作流会自动发布页面。

也可以在 Settings → Pages 中选择 **Deploy from a branch**，并将发布目录设为仓库根目录。

## 本地预览

直接打开 `index.html` 即可；或使用任意静态文件服务器预览。

## 内容来源

- 论文：Li, R., Zhang, J. & Zhao, X. *Communications Engineering* 5, 144 (2026).
- DOI：https://doi.org/10.1038/s44172-026-00684-7
- 代码与场景：https://github.com/lironui/SAR_WAKE
- SAR 源影像：European Space Agency (ESA)
