# SylphyZ.github.io

> 个人主页，托管于 GitHub Pages。

## 链接

- **个人主页** → [sylphyz.github.io](https://sylphyz.github.io)

## 仓库结构

```
SylphyZ.github.io/
├── index.html          # 主页
├── 404.html            # 友好错误页
├── games.html          # 小游戏
├── favicon.png         # 站点图标
├── og.png              # 社交分享预览图
├── robots.txt
├── sitemap.xml
├── scripts/
│   ├── render-notes.mjs # 将 Markdown 预渲染为可离线阅读的 HTML
│   └── check-site.mjs   # 检查元信息与站内链接
└── notebook/
    ├── index.html      # 笔记目录
    ├── publish.pdf     # 博士期间学习笔记 (PDF)
    ├── Bodie《Investments》/
    │   ├── Chapter5 ~ Chapter10 笔记 (.html)
    │   └── images/
    ├── Andrew Ang《Asset Management》/
    │   ├── Ch3, Ch6 ~ Ch8, Ch10, Ch14 笔记 (.html)
    │   └── images/
    └── 石川《因子投资方法与实践》/
        ├── 第1章 ~ 第7章及 GMM 专题笔记 (.html)
        └── images/
```

## 关于

本仓库为 [SylphyZ](https://github.com/SylphyZ) 的个人主页源码，内容包含个人简介、小游戏及学习笔记。

笔记内容涵盖：
- **博士期间学习笔记** — Flow Matching、扩散模型、SDE、Fourier 分析、高斯过程
- **Bodie《Investments》** — 风险收益、资本配置、分散化、CAPM、APT
- **Andrew Ang《Asset Management》** — 均值-方差投资、因子理论、因子投资
- **石川《因子投资方法与实践》** — 因子方法论、主流因子、多因子模型、异象研究与投资实践

## 本地维护

安装 Node.js 20 或更高版本后：

```bash
npm install
npm run build:notes
npm run check
```

`build:notes` 会从本地各文章同名的 `.md` 文件生成静态正文、图片懒加载属性和 `sitemap.xml`。Markdown 源文件由 `.gitignore` 排除，不上传至 GitHub；生成后的 HTML 可直接阅读，不依赖 Markdown CDN。
