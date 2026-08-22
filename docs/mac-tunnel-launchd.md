# Mac 隧道 launchd 常驻(一次配置,永不再管)

在 **Mac** 上执行以下三步。

## 1. 写 plist 文件

把下面内容存为 `~/Library/LaunchAgents/edu.fsu.hpg-tunnel.plist`
(把两处 `RAI_HOST` 换成你 ssh 登 rai 用的地址;若 Mac 上用户名对
rai 的 ssh 别名已配好,`ssh rai` 能通的话直接写 `rai`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>edu.fsu.hpg-tunnel</string>

  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/ssh</string>
    <string>-N</string>
    <string>-o</string><string>ServerAliveInterval=30</string>
    <string>-o</string><string>ServerAliveCountMax=3</string>
    <string>-o</string><string>ExitOnForwardFailure=yes</string>
    <string>-o</string><string>StrictHostKeyChecking=accept-new</string>
    <string>-R</string><string>2222:hpg.rc.ufl.edu:22</string>
    <string>xueqi@RAI_HOST</string>
  </array>

  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>15</integer>

  <key>StandardErrorPath</key>
  <string>/tmp/hpg-tunnel.err</string>
</dict>
</plist>
```

说明:直接用系统 ssh + `KeepAlive`,launchd 自己充当 autossh——进程
一退(网断、VPN 掉、rai 重启)15 秒后自动重拉,无需 autossh。

## 2. 加载并启动

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/edu.fsu.hpg-tunnel.plist
launchctl kickstart -k gui/$(id -u)/edu.fsu.hpg-tunnel
```

以后开机自动启动;VPN 重连后它会自己重试,不用再手动管。

## 3. 验证

Mac 上:`launchctl print gui/$(id -u)/edu.fsu.hpg-tunnel | grep state`
应显示 running;然后在 Discord 里说一声,我在 rai 侧确认 2222 通。

## 常用命令

- 停止:`launchctl bootout gui/$(id -u)/edu.fsu.hpg-tunnel`
- 看错误:`cat /tmp/hpg-tunnel.err`
- 前提:Mac→rai 的 ssh 免密(key)已配好(你平时能直接 ssh 就没问题);
  VPN 断开时它会循环重试,VPN 一回来隧道秒回。

## 注意

- 若 rai 上残留旧的 2222 监听导致 `remote port forwarding failed`,
  在 Discord 说一声,我在 rai 清理后它下一轮重试即成功。
- Mac 合盖睡眠仍会断(硬件限制);插电 + 系统设置里"防止自动进入
  睡眠"可避免,或接受"开盖即自动重连"。
