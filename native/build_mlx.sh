#!/bin/sh
# Build native/lev_mlx.c with mlx-c and mlx (native/mlx-c, cloned and
# built by the `jolt mlx` task at the pinned tag) into liblev_mlx.dylib for
# jolt run/test and liblev_mlx.a for jolt build, and put mlx.metallib next
# to the dylib: MLX loads its kernels from next to the binary that holds
# it (the dylib here; lev-server for a jolt build, so ship the metallib
# beside it), else from the path compiled in from this build tree. Run
# from the project root. mac only: MLX's GPU backend is Metal.
set -e
M=native/mlx-c
B=$M/build
TAG=$(cd "$M" && git describe --tags --abbrev=0 2>/dev/null || echo unknown)
MLX_SRC=$B/_deps/mlx-src
LIBS="$B/libmlxc.a $B/_deps/mlx-build/libmlx.a"
FMT=$(ls $B/_deps/fmt-build/libfmt*.a 2>/dev/null || true)
LIBS="$LIBS $FMT"
FRAMEWORKS="-framework Metal -framework Foundation -framework QuartzCore -framework Accelerate"
INC="-I$M -I$MLX_SRC -I$B/_deps/mlx-build"
case "$(uname -s)" in
  Darwin) ;;
  *) echo "build_mlx.sh: the MLX backend is mac only (Metal)"; exit 1 ;;
esac
cc -O2 -std=c11 -Wall -Wextra -DLEV_MLX_BUILD="\"mlx-c $TAG\"" $INC -c native/lev_mlx.c -o native/lev_mlx.o
c++ -dynamiclib native/lev_mlx.o $LIBS $FRAMEWORKS -o native/liblev_mlx.dylib
# one archive with every mlx member inside, for jolt build's force-load
libtool -static -o native/liblev_mlx.a native/lev_mlx.o $LIBS
rm -f native/lev_mlx.o
# the Metal kernels, colocated with the dylib
cp "$B/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib" native/mlx.metallib
# ld64 links a framework stub by lib<Name>.tbd in an -L dir: what deps.edn's
# :static {:lib ...} entries can express (build_llm.sh does the same)
sdk=$(xcrun --show-sdk-path)
mkdir -p native/frameworks
for f in Metal Foundation MetalKit Accelerate QuartzCore; do
  ln -sf "$sdk/System/Library/Frameworks/$f.framework/$f.tbd" "native/frameworks/lib$f.tbd"
done
echo "built: native/liblev_mlx.* + native/mlx.metallib (mlx-c $TAG)"
