#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./create_winboot.sh /path/to/windows.iso /dev/sdX1
#
# This version WILL:
# - wipe the USB disk
# - create an MBR (msdos) partition table
# - create + format a FAT32 partition
# - copy Windows ISO contents (excluding sources/install.wim)
# - split install.wim into install.swm parts (FAT32-friendly)
# - sync just the USB mount
# - unmount and power-off the USB device

ISO_PATH="${1:-}"
USB_PART="${2:-}"

if [[ -z "$ISO_PATH" || -z "$USB_PART" ]]; then
  echo "Usage: $0 /path/to/windows.iso /dev/sdX1" >&2
  exit 2
fi

if [[ ! -f "$ISO_PATH" ]]; then
  echo "ISO not found: $ISO_PATH" >&2
  exit 1
fi

for cmd in rsync wimlib-imagex parted mkfs.fat lsblk udevadm; do
  command -v "$cmd" >/dev/null || { echo "Missing dependency: $cmd" >&2; exit 1; }
done

# Derive parent disk from partition (e.g., /dev/sda1 -> /dev/sda)
PKNAME="$(lsblk -no PKNAME "$USB_PART" 2>/dev/null || true)"
if [[ -z "$PKNAME" ]]; then
  echo "Could not derive parent disk for: $USB_PART" >&2
  exit 1
fi
USB_DISK="/dev/$PKNAME"

# Safety: refuse if this doesn't look like a removable disk
REMOVABLE_PATH="/sys/block/$PKNAME/removable"
if [[ ! -r "$REMOVABLE_PATH" || "$(cat "$REMOVABLE_PATH")" != "1" ]]; then
  echo "Refusing to wipe $USB_DISK (does not look removable)." >&2
  echo "If this is actually your USB stick, check lsblk and pass the correct /dev/sdX1." >&2
  exit 1
fi

# Safety: refuse if any partition on this disk is mounted at /
if lsblk -nr -o MOUNTPOINT "$USB_DISK" | grep -qx '/'; then
  echo "Refusing to wipe $USB_DISK (it contains your root filesystem /)." >&2
  exit 1
fi

# start the USB creation for Windows
sudo mkdir -p /mnt/iso /mnt/usb

mounted_iso=0
mounted_usb=0

cleanup() {
  # Flush only the USB mount (avoid global sync)
  if mountpoint -q /mnt/usb; then
    sync -f /mnt/usb || true
  fi

  if (( mounted_usb )); then
    sudo umount /mnt/usb || true
  fi
  if (( mounted_iso )); then
    sudo umount /mnt/iso || true
  fi
}
trap cleanup EXIT

echo ">>> WIPING and repartitioning $USB_DISK to MBR/msdos partition table..."
# Unmount anything auto-mounted from that disk
while read -r mp; do
  [[ -n "$mp" ]] && sudo umount "$mp" || true
done < <(lsblk -nr -o MOUNTPOINT "$USB_DISK" | awk 'NF')

# Create MBR partition table + one FAT32 partition
sudo parted -s "$USB_DISK" mklabel msdos # this initializes the disk with an MBR (Master Boot Record) partition table
sudo parted -s "$USB_DISK" mkpart primary fat32 1MiB 100% # creates a primary partition. `fat32` Tells the system the intended filesystem type (Note: `parted` doesn't actually format the drive here; it just sets the partition ID). `1MiB 100%` defines the start and end of the partition (Starting at 1MiB is a modern standard for alignment, ensuring the disk performs efficiently; 100% tells it to use all remaining space on the drive)
sudo parted -s "$USB_DISK" set 1 boot on # selects the first partition (1) and sets the boot flag to "on."

sudo udevadm settle # tells the script to wait until the system has finished reacting to the previous commands. When you run parted to create or change partitions, the Linux kernel notices the changes and triggers "events" (like creating new device nodes in /dev/sdb1). Because these events happen in the background (asynchronously), your script might try to run the next command before the drive is actually ready.

# Re-detect the new first partition path robustly
USB_PART_NEW="/dev/$(lsblk -ln -o NAME "$USB_DISK" | awk 'NR==2{print $1}')"
if [[ -z "$USB_PART_NEW" || ! -b "$USB_PART_NEW" ]]; then
  echo "Could not find created partition on $USB_DISK" >&2
  exit 1
fi

echo ">>> Formatting the filesystem of $USB_PART_NEW as FAT32 (labelled WIN11) ..."
sudo mkfs.fat -F 32 -n WIN11 "$USB_PART_NEW"

echo ">>> Mounting ISO + USB in /mnt/"
if ! mountpoint -q /mnt/iso; then
  sudo mount -o loop,ro "$ISO_PATH" /mnt/iso
  mounted_iso=1
fi

if ! mountpoint -q /mnt/usb; then
  sudo mount "$USB_PART_NEW" /mnt/usb
  mounted_usb=1
fi

# Copy ISO contents to USB, excluding install.wim and avoiding chown/chgrp/chmod issues
echo ">>> Copying /mnt/iso (excluding sources/install.wim) to /mnt/usb"
sudo rsync -rltvh \
  --exclude='sources/install.wim' \
  --no-owner --no-group --no-perms \
  --modify-window=1 \
  /mnt/iso/ /mnt/usb/

# Split install.wim into FAT32-friendly chunks
echo ">>> Splitting install.wim into .swm ..."
sudo mkdir -p /mnt/usb/sources
sudo rm -f /mnt/usb/sources/install*.swm
sudo wimlib-imagex split /mnt/iso/sources/install.wim /mnt/usb/sources/install.swm 3800

# Your requested line (sync only this mount)
echo ">>> sync -f /mnt/usb ..."
sync -f /mnt/usb

echo ">>> Done copying. Unmounting..."
sudo umount /mnt/usb
mounted_usb=0
sudo umount /mnt/iso
mounted_iso=0

# Your requested power-off (but automatic disk)
if command -v udisksctl >/dev/null; then
  echo ">>> Powering off $USB_DISK ..."
  sudo udisksctl power-off -b "$USB_DISK"
else
  echo "NOTE: udisksctl not found; skipping power-off." >&2
fi

echo ">>> removing /mnt/iso and /mnt/usb dirs"
sudo rm -rf /mnt/iso /mnt/usb

echo "Done. Replug the USB if you want to browse it again, then boot from it (UEFI boot menu)."

