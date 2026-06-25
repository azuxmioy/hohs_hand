#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.10.13}"
PREFIX="${PYTHON_PREFIX:-/data/hohs2/python/cpython-$PYTHON_VERSION}"
BUILD_ROOT="${PYTHON_BUILD_ROOT:-/data/hohs2/src}"
DEPS_PREFIX="${PYTHON_DEPS_PREFIX:-/data/hohs2/opt/python-build-deps}"
MODULES="${MODULES:-gcc/11 cuda/11.8 cudnn/8.4_cuda11.x}"
BZIP2_VERSION="${BZIP2_VERSION:-1.0.8}"
SQLITE_AUTOCONF="${SQLITE_AUTOCONF:-3450300}"
SQLITE_YEAR="${SQLITE_YEAR:-2024}"
if [[ -d /data/hohs2 ]]; then
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/hohs2/pip-cache}"
fi

if ! type module >/dev/null 2>&1; then
  for init_script in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
    if [[ -f "$init_script" ]]; then
      # shellcheck disable=SC1090
      source "$init_script"
      break
    fi
  done
fi

if type module >/dev/null 2>&1 && [[ -n "$MODULES" ]]; then
  module load $MODULES
fi

python_has_required_modules() {
  "$PREFIX/bin/python3.10" - <<'PY'
import bz2
import sqlite3
PY
}

if [[ -x "$PREFIX/bin/python3.10" && "${FORCE_REBUILD:-0}" != "1" ]]; then
  if python_has_required_modules; then
    "$PREFIX/bin/python3.10" --version
    exit 0
  fi
  echo "Existing Python is missing required stdlib modules; rebuilding."
fi

download() {
  local url="$1"
  local dest="$2"
  if [[ ! -f "$dest" ]]; then
    if command -v curl >/dev/null 2>&1; then
      curl -L "$url" -o "$dest"
    elif command -v wget >/dev/null 2>&1; then
      wget "$url" -O "$dest"
    else
      echo "Need curl or wget to download $url"
      exit 1
    fi
  fi
}

mkdir -p "$BUILD_ROOT" "$(dirname "$PREFIX")" "$DEPS_PREFIX"
BZIP2_PREFIX="$DEPS_PREFIX/bzip2-$BZIP2_VERSION"
SQLITE_PREFIX="$DEPS_PREFIX/sqlite-$SQLITE_AUTOCONF"

if [[ ! -f "$BZIP2_PREFIX/include/bzlib.h" || ! -f "$BZIP2_PREFIX/lib/libbz2.a" ]]; then
  BZIP2_TARBALL="$BUILD_ROOT/bzip2-$BZIP2_VERSION.tar.gz"
  download "https://sourceware.org/pub/bzip2/bzip2-$BZIP2_VERSION.tar.gz" "$BZIP2_TARBALL"
  rm -rf "$BUILD_ROOT/bzip2-$BZIP2_VERSION"
  tar -xzf "$BZIP2_TARBALL" -C "$BUILD_ROOT"
  make -C "$BUILD_ROOT/bzip2-$BZIP2_VERSION" -j"$(nproc)" CFLAGS="-fPIC -O2 -D_FILE_OFFSET_BITS=64"
  make -C "$BUILD_ROOT/bzip2-$BZIP2_VERSION" PREFIX="$BZIP2_PREFIX" install
fi

if [[ ! -f "$SQLITE_PREFIX/include/sqlite3.h" || ! -f "$SQLITE_PREFIX/lib/libsqlite3.so" ]]; then
  SQLITE_TARBALL="$BUILD_ROOT/sqlite-autoconf-$SQLITE_AUTOCONF.tar.gz"
  download "https://www.sqlite.org/$SQLITE_YEAR/sqlite-autoconf-$SQLITE_AUTOCONF.tar.gz" "$SQLITE_TARBALL"
  rm -rf "$BUILD_ROOT/sqlite-autoconf-$SQLITE_AUTOCONF"
  tar -xzf "$SQLITE_TARBALL" -C "$BUILD_ROOT"
  cd "$BUILD_ROOT/sqlite-autoconf-$SQLITE_AUTOCONF"
  ./configure --prefix="$SQLITE_PREFIX" --disable-readline CFLAGS="-O2 -fPIC"
  make -j"$(nproc)"
  make install
fi

TARBALL="$BUILD_ROOT/Python-$PYTHON_VERSION.tgz"
SRC_DIR="$BUILD_ROOT/Python-$PYTHON_VERSION"
URL="https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz"

download "$URL" "$TARBALL"

rm -rf "$SRC_DIR"
tar -xzf "$TARBALL" -C "$BUILD_ROOT"
cd "$SRC_DIR"
export CPPFLAGS="-I$BZIP2_PREFIX/include -I$SQLITE_PREFIX/include ${CPPFLAGS:-}"
export LDFLAGS="-L$BZIP2_PREFIX/lib -L$SQLITE_PREFIX/lib -Wl,-rpath,$SQLITE_PREFIX/lib ${LDFLAGS:-}"
export LD_LIBRARY_PATH="$BZIP2_PREFIX/lib:$SQLITE_PREFIX/lib:${LD_LIBRARY_PATH:-}"
./configure --prefix="$PREFIX" --with-ensurepip=install --enable-loadable-sqlite-extensions
make -j"$(nproc)"
make install
"$PREFIX/bin/python3.10" --version
python_has_required_modules
