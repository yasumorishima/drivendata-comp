---
name: Colab experiment automation setup
description: RPi5 xrdp + Google Drive連携のKaggle実験環境。セットアップ手順と復帰方法
type: reference
originSessionId: bb0a5dc1-f2e0-418e-bd85-d63b3389b92f
---
# Colab実験自動化環境（2026-03-18更新）

## 構成（全プロジェクト共通: Kaggle / DrivenData / SIGNATE / 地震研究等）
- **ローカルPC**: Claude CodeがDrive上のEXPフォルダにconfig/train.pyを書く
- **Google Drive for Desktop**: `G:/マイドライブ/kaggle/` で同期
- **RPi5 xrdp**: リモートデスクトップ（切断してもセッション維持）
- **Colab**: ファイル監視ノートブック（while Trueループ、新config検知→train.py自動実行）
- **CDP keepalive**: 全コンペのColab GPU実験で共通利用（DrivenData専用ではない）

## Colab URL
- **v4（修正済み）**: Driveの `kaggle/runner/experiment_runner_v4.ipynb` を開く
- **旧v3**: <Colab notebook URL (removed)>

## セッション開始手順

### Claude Codeがやること
```bash
# RDPを立ち上げるだけ
bash "C:/Users/fw_ya/Desktop/Claude_code/scripts/start-colab.sh"
```

### ユーザーがやること
1. RDP内で右クリック → ブラウザ起動
2. Colabノートブックを開く → T4 GPU選択 → セル実行
3. 作業が終わったらRDPウィンドウを閉じる（セッションは維持される）

## RDP設定
- **設定ファイル**: `C:\Users\fw_ya\Desktop\rpi5.rdp`（資格情報保存済み）
- **RPi5側**: xrdp + openbox（右クリックでメニュー）
- **ユーザー/パスワード**: yasu / colab2026
- **ポート**: 3389（ufw許可済み）

## 実験修正→再実行の手順（Claude Codeの作業フロー）
1. **ファイル修正**: train.py / config yaml / requirements.txt をDrive上で直接編集
2. **result.json削除**: 失敗した実験の `output/child-expXXX/result.json` を削除（runnerが再検出するため）
3. **RDP起動**: `bash "C:/Users/fw_ya/Desktop/Claude_code/scripts/start-colab.sh"`
4. **ユーザーに依頼**: RDP内でColabノートブックを開いてrunner再実行してもらう
5. **結果確認**: Drive上の `result.json` / `train.log` を読む（SSH不要、Drive for Desktopで同期される）

## requirements.txt の配置ルール
runnerの `install_comp_deps` は以下の順で探す:
1. `exp_dir.parent / "requirements.txt"` → 例: `pasketti/EXP/requirements.txt`
2. `exp_dir / "requirements.txt"` → 例: `pasketti/EXP/EXP001/requirements.txt`
→ **`comp_dir/requirements.txt`（例: `pasketti/requirements.txt`）は探さない**ので、`EXP/requirements.txt` にもコピーを置く

## CDP keepalive（自動セッション維持）
- **方式**: Chromium `--remote-debugging-port=9222 --remote-allow-origins=*` + systemd Python CDP スクリプト
- **間隔**: 20分（90分タイムアウトに十分余裕あり）
- **スクリプト**: `/home/yasu/colab-keepalive/cdp_keepalive.py`
- **サービス**: `colab-keepalive.service`（systemd user）
- **openbox autostart**: `/home/yasu/.config/openbox/autostart` でRDP接続時にChromium自動起動
- **初回のみ**: RDP内ターミナルで `bash ~/colab-keepalive/start-chromium-colab.sh`、次回以降はautostart
- **ハートビート**: runnerが `G:/マイドライブ/kaggle/runner_heartbeat.json` を更新
- **確認方法**: `journalctl --user -u colab-keepalive.service --since '30 min ago'` or heartbeat.json
- **注意**: SSH経由ではxrdpセッションのChromiumをkillできない（権限問題）。RDP内ターミナルから操作する
- **DISPLAY**: xrdpセッションのDISPLAYは接続ごとに変わる可能性あり（:10, :11等）。start-chromium-colab.shが自動検出

## colab-mcp（稼働中 2026-04-24 Windows PC で動作確認完了）

**Google公式MCPサーバー**。Claude CodeからColab GPUを操作。FastMCP 2.14.5 / stdio + WebSocket proxy。

### 動作モード
- **proxy mode**（デフォルト `-p` 有効）: ブラウザで Colab タブ開く → WebSocket (`ws://localhost:54455`) で bridge → Colab 側 runtime でコード実行
- **runtime mode**（`-r`）: headless 直接 API。現在 Google 内部専用 scope で外部不可、公開待ち

### 設定場所（重要：いつも混乱するのでここ固定）
- **Windows PC**: `C:/Users/fw_ya/.mcp.json`（project-level、`~/.claude/settings.json` でも `~/.claude.json` でもない）
- **RPi5**: `~/.claude/settings.json`
- Windows の uvx 実体: `C:\Users\fw_ya\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\uvx.exe`（Microsoft Store 版 Python 3.12 付属）
- Log: `C:/Users/fw_ya/AppData/Local/Temp/colab-mcp-logs/colab-mcp.<timestamp>.log`

### Windows .mcp.json（稼働中）
```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "C:\\Users\\fw_ya\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\Scripts\\uvx.exe",
      "args": ["git+https://github.com/googlecolab/colab-mcp", "-l", "C:\\Users\\fw_ya\\AppData\\Local\\Temp\\colab-mcp-logs"],
      "env": { "UV_HTTP_TIMEOUT": "120" },
      "timeout": 120000
    }
  }
}
```

### RPi5 settings.json
```json
"colab-mcp": {
  "command": "/home/yasu/.local/bin/uvx",
  "args": ["--python", "3.13", "git+https://github.com/googlecolab/colab-mcp"],
  "env": { "UV_HTTP_TIMEOUT": "120" },
  "timeout": 120000
}
```

### `/mcp` が "Failed to reconnect" になったとき
- Claude Code 側の connection state 劣化。config も server バイナリも健全でも出る（[claude-code#34744](https://github.com/anthropics/claude-code/issues/34744) known issue、診断情報ゼロ）
- **解決: Claude Code セッション再起動**。新セッションで `/mcp` 実行 → "Reconnected to colab-mcp" 表示で fresh 接続完了（2026-04-24 実証）
- 再起動後は deferred tool として以下が使える:
  - `mcp__colab-mcp__open_colab_browser_connection`
  - `ListMcpResourcesTool` / `ReadMcpResourceTool`
- proxy mode は **ブラウザで Colab タブを開く必要** がある。`open_colab_browser_connection` はそれを誘導する

### トラブルシュート手順（2026-04-24 実施）
1. uvx パス確認: PowerShell で `& $uvx --version` → `uvx 0.8.4` 表示されれば OK（Bash で絶対パス .exe 叩くと 10 分+ hang するので **必ず PowerShell tool 経由**、`block-bash-windows-exe.sh` hook で強制）
2. colab-mcp 単体起動: PowerShell で `Start-Process $uvx -ArgumentList 'git+...','-l',$logdir -RedirectStandardOutput ... -RedirectStandardError ...`（file-redirect 必須、pipe 直読は deadlock）
3. log 確認: `$env:TEMP\colab-mcp-logs\*.log` に `server listening on 127.0.0.1:54455` が出れば server 側 OK
4. Claude Code 再起動 → `/mcp` で "Reconnected" 出るまで繰り返し

## 注意（旧方式: RPi5 + Drive同期）
- Chromiumを閉じるとColabセッションが切れる（RDP切断はOK）
- 12時間でColab最大セッション制限 → 再度開き直す
- ノートブック修正はDrive上のファイルを編集 → 新しいファイル名で保存（Colabはキャッシュするため）

## GH_TOKEN（Colab→GitHub Artifact DL用）
- `G:\マイドライブ\kaggle\.gh_token` にファイル配置
- setup_data.py が Colab Secrets → env var → Driveファイル の順でフォールバック読み込み

## Drive上のEXP構成
```
G:/マイドライブ/kaggle/
├── runner/
│   ├── experiment_runner.ipynb      # v3（古い、total_memバグあり）
│   └── experiment_runner_v4.ipynb   # v4（修正済み、こちらを使う）
├── .gh_token                        # GitHub PAT（Artifact DL用）
├── .wandb_api_key                   # W&B API Key（本番学習用）
├── s6e3-churn/EXP/...               # Kaggle S6E3（完了）
├── pasketti/EXP/...                 # DrivenData Phonetic Track
├── deep-past/EXP/...                # Kaggle Deep Past
├── rna-3d-folding/EXP/...           # Kaggle RNA
├── signate-nir-moisture/EXP/...     # SIGNATE NIR Moisture
├── geohazard/EXP/...               # 地震研究（ML学習が必要な場合）
└── <新規プロジェクト>/EXP/...       # 同じ構成で追加するだけ（コンペ以外もOK）
```

## openclaw-gateway
停止中（メモリ370MB節約）。復帰: `systemctl --user enable --now openclaw-gateway`

---

**2026-08-26 に `_archive/` へ移動。** Colab 前提の実験手順。現行の計算配置は [[feedback_compute-placement]]（C++=Codespace / 軽作業=RPi5）が正。
