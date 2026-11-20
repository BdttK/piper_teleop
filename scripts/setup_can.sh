#!/usr/bin/env bash
# 按 USB 口(bus-info)把 4 个 candleLight CAN 适配器固定命名并激活(1Mbps):
#   can_leader_left / can_leader_right / can_follower_left / can_follower_right
#
# 使用步骤:
#   1. 先只插一个适配器,执行: sudo ethtool -i canX | grep bus-info
#      记录 bus-info(如 1-12:1.0);逐个插入并记录,填写下面 USB_PORTS 映射。
#      也可先运行 `bash scripts/setup_can.sh --list` 列出当前所有 CAN 接口的 bus-info。
#   2. 插好全部 4 个适配器后执行: sudo bash scripts/setup_can.sh
#
# 注意: 接口名对应物理 USB 口,插拔换口后需重新执行本脚本。
set -euo pipefail

declare -A USB_PORTS
# bus-info -> "接口名:波特率"
USB_PORTS["1-12:1.0"]="can_leader_left:1000000"
USB_PORTS["1-13:1.0"]="can_follower_left:1000000"
USB_PORTS["1-14:1.0"]="can_leader_right:1000000"
USB_PORTS["1-15:1.0"]="can_follower_right:1000000"
DEFAULT_BITRATE=1000000

list_interfaces() {
    for iface in $(ip -br link show type can | awk '{print $1}'); do
        bus=$(ethtool -i "$iface" 2>/dev/null | awk '/bus-info/ {print $2}')
        echo "  $iface  bus-info=${bus:-未知}"
    done
}

if [ "${1:-}" = "--list" ]; then
    echo "当前 CAN 接口:"
    list_interfaces
    echo
    echo "把 bus-info 填到脚本顶部的 USB_PORTS 映射中。"
    exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "需要 root 权限: sudo bash $0" >&2
    exit 1
fi

modprobe gs_usb 2>/dev/null || true

ifaces=$(ip -br link show type can | awk '{print $1}')
if [ -z "$ifaces" ]; then
    echo "未检测到 CAN 接口(检查 gs_usb 驱动与 USB 连接)" >&2
    exit 1
fi

for iface in $ifaces; do
    bus=$(ethtool -i "$iface" 2>/dev/null | awk '/bus-info/ {print $2}')
    if [ -z "$bus" ]; then
        echo "[跳过] $iface: 无法获取 bus-info"
        continue
    fi
    mapping="${USB_PORTS[$bus]:-}"
    if [ -z "$mapping" ]; then
        echo "[未映射] $iface (bus-info=$bus) 不在 USB_PORTS 中,跳过"
        continue
    fi
    IFS=':' read -r target bitrate <<< "$mapping"
    bitrate="${bitrate:-$DEFAULT_BITRATE}"

    ip link set "$iface" down
    if [ "$iface" != "$target" ]; then
        ip link set "$iface" name "$target"
        iface="$target"
    fi
    ip link set "$iface" type can bitrate "$bitrate"
    ip link set "$iface" up
    echo "[完成] bus-info=$bus -> $iface @ ${bitrate}bps"
done

echo
echo "当前 CAN 接口:"
list_interfaces
