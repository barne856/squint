FROM mcr.microsoft.com/devcontainers/base:jammy

# C++ development environment, uses clang, libc++, and ninja
ENV CC=clang
ENV CXX=clang++
ENV LDFLAGS="-stdlib=libc++ -fuse-ld=lld"
ENV CXXFLAGS=-stdlib=libc++
ENV CMAKE_GENERATOR=Ninja

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    # C++ tools
    clang \
    clang-format \
    clang-tidy \
    clangd \
    libc++-dev \
    libc++abi-dev \
    lldb \
    lld \
    ninja-build \
    # BLAS and LAPACKE
    libopenblas-dev \
    liblapacke-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
