#!/bin/bash
# Build the native filters with the portable GCC (no install, no admin):
#   native/build/filmdust.node      Node-API module for tests (node test/native_parity.js)
#   native/build/filmdust.uxpaddon  Photoshop hybrid addon (win/x64)
# -ffp-contract=off and no fast-math: every operation rounds exactly as V8's does.
set -e
cd "$(dirname "$0")"
# GCC: a MinGW-w64 bin folder (winlibs works, no install needed) when g++ is not already on the PATH
[ -n "$GCC" ] && export PATH="$GCC:$PATH"
mkdir -p build
FLAGS="-O3 -std=c++17 -march=x86-64-v3 -ffp-contract=off -fno-fast-math -fexcess-precision=standard -Wall -Wno-unused-function"
[ -f build/libnode_api.a ] || dlltool -d node-api/node_api.def -l build/libnode_api.a -D node.exe
g++ $FLAGS -shared -o build/filmdust.node -Inode-api src/node_glue.cpp src/filters.cpp src/dustnet.cpp src/maps.cpp src/regions.cpp src/pool.cpp src/cues.cpp src/soft.cpp src/blobs.cpp src/thin.cpp -Lbuild -lnode_api -static -static-libgcc -static-libstdc++ -pthread
[ -f uxp-sdk/utilities/UxpAddon.cpp ] || { echo "native/uxp-sdk is empty: put Adobe's UXP hybrid plugin SDK headers there (see native/uxp-sdk/README.md); filmdust.node is built"; exit 1; }
g++ $FLAGS -shared -o build/filmdust.uxpaddon src/uxp_glue.cpp src/filters.cpp src/dustnet.cpp src/maps.cpp src/regions.cpp src/pool.cpp src/cues.cpp src/soft.cpp src/blobs.cpp src/thin.cpp uxp-sdk/utilities/UxpAddon.cpp -static -static-libgcc -static-libstdc++ -pthread
ls -la build/*.node build/*.uxpaddon
