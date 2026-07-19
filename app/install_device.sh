#!/bin/bash
# Build a signed Release of MPNNBench and install it on the connected iPhone.
# Prerequisite (one-time): add the Apple ID javiercv@uw.edu to Xcode
#   Xcode ▸ Settings ▸ Accounts ▸ + ▸ Apple ID
# so automatic signing (team VT99UQUQ89) can create the development profile.
set -e
cd "$(dirname "$0")"

DEVID=00008130-00040D8C01F0001C          # Javier's iPhone 15 Pro (iOS 26.5.2)
DEVCTL_ID=F691F4CA-333B-5669-830F-2B400CEE3993

echo "▸ Regenerating project…"
xcodegen generate >/dev/null

echo "▸ Building signed Release for device…"
xcodebuild -project MPNNBench.xcodeproj -scheme MPNNBench \
  -configuration Release \
  -destination "platform=iOS,id=$DEVID" \
  -skipPackagePluginValidation -skipMacroValidation \
  -allowProvisioningUpdates \
  -derivedDataPath build_device \
  DEVELOPMENT_TEAM=VT99UQUQ89 CODE_SIGN_STYLE=Automatic \
  build

APP="build_device/Build/Products/Release-iphoneos/MPNNBench.app"
echo "▸ Installing $APP …"
xcrun devicectl device install app --device "$DEVCTL_ID" "$APP"

echo "✓ Installed. On the phone: Settings ▸ General ▸ VPN & Device Management ▸ trust"
echo "  the 'Apple Development: javiercv@uw.edu' certificate, then launch MPNNBench."
