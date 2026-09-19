#!/bin/sh
# Build native/lev_llm.c with llama.cpp (native/llama.cpp, cloned by the
# `jolt llama` task at the pinned tag) into liblev_llm.{dylib,so} for jolt
# run/test and liblev_llm.a for jolt build. Run from the project root.
set -e
L=native/llama.cpp
TAG=$(cd "$L" && git describe --tags --abbrev=0 2>/dev/null || echo unknown)
LIBS="$L/build/src/libllama.a $L/build/ggml/src/libggml.a $L/build/ggml/src/libggml-cpu.a $L/build/ggml/src/libggml-base.a"
case "$(uname -s)" in
  Darwin)
    LIBS="$LIBS $L/build/ggml/src/ggml-metal/libggml-metal.a"
    FRAMEWORKS="-framework Metal -framework Foundation -framework MetalKit -framework Accelerate"
    cc -O2 -std=c11 -Wall -Wextra -DLEV_LLAMA_BUILD="\"$TAG\"" -I"$L/include" -I"$L/ggml/include" \
       -dynamiclib native/lev_llm.c $LIBS $FRAMEWORKS -lc++ -o native/liblev_llm.dylib
    cc -O2 -std=c11 -c -DLEV_LLAMA_BUILD="\"$TAG\"" -I"$L/include" -I"$L/ggml/include" native/lev_llm.c -o native/lev_llm.o
    # one archive with every llama.cpp member inside, for jolt build's force-load
    libtool -static -o native/liblev_llm.a native/lev_llm.o $LIBS
    # ld64 links a framework stub by lib<Name>.tbd in an -L dir: what deps.edn's
    # :static {:lib ...} entries can express
    sdk=$(xcrun --show-sdk-path)
    mkdir -p native/frameworks
    for f in Metal Foundation MetalKit Accelerate; do
      ln -sf "$sdk/System/Library/Frameworks/$f.framework/$f.tbd" "native/frameworks/lib$f.tbd"
    done
    ;;
  *)
    cc -O2 -std=c11 -Wall -Wextra -fPIC -DLEV_LLAMA_BUILD="\"$TAG\"" -I"$L/include" -I"$L/ggml/include" \
       -shared native/lev_llm.c -Wl,--whole-archive $LIBS -Wl,--no-whole-archive -lstdc++ -lm -lpthread -o native/liblev_llm.so
    cc -O2 -std=c11 -c -fPIC -DLEV_LLAMA_BUILD="\"$TAG\"" -I"$L/include" -I"$L/ggml/include" native/lev_llm.c -o native/lev_llm.o
    printf 'create native/liblev_llm.a\naddmod native/lev_llm.o\n' > native/llm.mri
    for a in $LIBS; do printf 'addlib %s\n' "$a" >> native/llm.mri; done
    printf 'save\nend\n' >> native/llm.mri
    ar -M < native/llm.mri
    rm -f native/llm.mri
    ;;
esac
rm -f native/lev_llm.o
echo "built: native/liblev_llm.* ($TAG)"
