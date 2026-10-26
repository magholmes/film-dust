#!/bin/bash
# Build the native filters:
#   native/build/filmdust.node      Node-API module for the tests (node test/native_fuzz.js), on any platform
#   native/build/filmdust.uxpaddon  Photoshop hybrid addon (Windows x64; needs Adobe's UXP SDK headers in native/uxp-sdk)
# Windows: a MinGW-w64 GCC (winlibs works, no install needed), on the PATH or its bin folder in $GCC.
# Linux and macOS: the system's g++ / clang++, for the tests.
# -ffp-contract=off and no fast-math: every operation rounds exactly as V8's does.
# -march=x86-64: runs on every 64-bit PC (an AVX2-only build crashed older and budget processors). Where AVX2
# pays, a kernel has a second copy built for it and picked at run time (src/cpu.h).
set -e
cd "$(dirname "$0")"
[ -n "$GCC" ] && export PATH="$GCC:$PATH"
mkdir -p build
SRC="src/filters.cpp src/dustnet.cpp src/maps.cpp src/regions.cpp src/pool.cpp src/cues.cpp src/soft.cpp src/blobs.cpp src/thin.cpp"
COMMON="-O3 -std=c++17 -ffp-contract=off -fno-fast-math -Wall -Wno-unused-function"
X86=""
[ "$(uname -m)" = x86_64 ] && X86="-march=x86-64 -mtune=generic"
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*)
    FLAGS="$COMMON -march=x86-64 -mtune=generic -fexcess-precision=standard"
    [ -f build/libnode_api.a ] || dlltool -d node-api/node_api.def -l build/libnode_api.a -D node.exe
    g++ $FLAGS -shared -o build/filmdust.node -Inode-api src/node_glue.cpp $SRC -Lbuild -lnode_api -static -static-libgcc -static-libstdc++ -pthread
    if [ -f uxp-sdk/utilities/UxpAddon.cpp ]; then
      g++ $FLAGS -shared -o build/filmdust.uxpaddon src/uxp_glue.cpp $SRC uxp-sdk/utilities/UxpAddon.cpp -static -static-libgcc -static-libstdc++ -pthread
    else
      echo "native/uxp-sdk is empty: put Adobe's UXP hybrid plugin SDK headers there (see native/uxp-sdk/README.md); filmdust.node is built"
    fi
    ;;
  Darwin)
    clang++ $COMMON $X86 -bundle -undefined dynamic_lookup -o build/filmdust.node -Inode-api src/node_glue.cpp $SRC -pthread
    ;;
  *)
    g++ $COMMON $X86 -fexcess-precision=standard -fPIC -shared -o build/filmdust.node -Inode-api src/node_glue.cpp $SRC -pthread
    ;;
esac
ls -la build/filmdust.node build/filmdust.uxpaddon 2>/dev/null || true
