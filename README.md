# 🎓 校园知识库助手 (Campus AI Assistant)

这是一个基于 RAG (检索增强生成) 的智能助手，支持文档问答和 AI 绘图。

## ✨ 功能特点
1. **📚 文档问答**：上传 PDF/TXT/DOCX 文档，基于文档内容进行智能回答。
2. **🧠 深度思考**：集成 DeepSeek R1 模型，支持复杂逻辑推理。
3. **🎨 创意绘图**：
   - OpenAI DALL-E 3
   - Bing Image Creator (免费)
   - SiliconFlow Flux.1 (免费/高质量)
4. **☁️ 云端备份**：支持自动将上传的文件备份到阿里云 OSS。

---

## 🚀 如何运行

### 方法 1：直接运行 (推荐)
我们为您准备了自动启动脚本，双击即可运行。

- **Windows 用户**: 双击 `启动(Windows).bat`
- **Mac / Linux 用户**: 在终端运行 `./启动(Mac_Linux).sh`

*(脚本会自动创建虚拟环境并安装所需的依赖库)*

### 方法 2：手动安装
如果您熟悉 Python，也可以手动运行：

1. **创建虚拟环境**:
   ```bash
   python -m venv venv
   # Windows 激活
   .\venv\Scripts\activate
   # Mac/Linux 激活
   source venv/bin/activate
   ```

2. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

3. **运行应用**:
   ```bash
   streamlit run app.py
   ```

---

## 🌐 在线部署 (如何让别人通过网址访问)

如果您希望别人不需要下载代码，直接通过网址访问，可以使用 **Streamlit Community Cloud** (免费)。

1. 将此项目代码上传到 **GitHub**。
2. 访问 [share.streamlit.io](https://share.streamlit.io/) 并使用 GitHub 账号登录。
3. 点击 "New app"，选择您的仓库和 `app.py` 文件。
4. 点击 "Deploy" 部署。
5. 部署成功后，您会获得一个网址 (例如 `https://your-project.streamlit.app`)，分享给他人即可。

> **注意**: 
> - 在线部署时，SQLite 数据库 (`users.db`) 在重启后会被重置。如果需要持久化保存用户数据，建议后续升级为云数据库 (如 Supabase)。
> - Bing Image Creator 在云端服务器可能会因为 IP 问题无法使用，建议优先使用 SiliconFlow 或 OpenAI。
