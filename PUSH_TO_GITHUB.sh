#!/bin/bash
# GitHub推送脚本
# 使用方法: bash PUSH_TO_GITHUB.sh

echo "================================================"
echo "准备推送到GitHub"
echo "================================================"
echo ""

# 检查当前目录
if [ ! -d ".git" ]; then
    echo "错误：当前不在Git仓库目录中"
    exit 1
fi

# 显示当前状态
echo "当前Git状态："
git status --short | head -10
echo ""

# 显示将要推送的提交
echo "将要推送的提交："
git log origin/master..HEAD --oneline 2>/dev/null || git log --oneline -3
echo ""

# 确认推送
read -p "确认推送到GitHub? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消推送"
    exit 0
fi

# 尝试推送
echo ""
echo "正在推送..."
git push origin master

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✅ 推送成功！"
    echo "================================================"
    echo ""
    echo "查看你的仓库："
    echo "https://github.com/liziyu6666/NTU-Dissertation"
    echo ""
else
    echo ""
    echo "================================================"
    echo "❌ 推送失败"
    echo "================================================"
    echo ""
    echo "请查看 organized/docs/GITHUB_PUSH_GUIDE.md"
    echo "或尝试："
    echo "  1. 配置SSH密钥"
    echo "  2. 使用Personal Access Token"
    echo ""
fi
