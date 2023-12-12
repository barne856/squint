FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-22.04

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.28.0-rc5"
ARG NINJA_VERSION="1.11.1"
ARG LLVM_VERSION="17"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

# Download and install Ninja
RUN wget https://github.com/ninja-build/ninja/archive/v${NINJA_VERSION}.tar.gz && \
    tar xf v${NINJA_VERSION}.tar.gz && \
    cd ninja-${NINJA_VERSION} && \
    cmake -Bbuild-cmake -H. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build-cmake && \
    mv build-cmake/ninja /usr/local/bin/ninja && \
    cd .. && \
    rm -rf v${NINJA_VERSION}.tar.gz ninja-${NINJA_VERSION}

# Build patched clangd from source
RUN git clone --depth 1 --branch clangdmodules https://github.com/ChuanqiXu9/llvm-project.git && \
    cd llvm-project && \
    mkdir build && \
    cd build && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lldb;lld" -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind" -DCMAKE_INSTALL_PREFIX="/usr" ../llvm && \
    cmake --build . --target install && \
    cd ../.. && \
    rm -rf llvm-project

# Download and install Intel MKL
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB -O /tmp/intel-key.pub && \
    gpg --no-default-keyring --keyring /etc/apt/trusted.gpg.d/intel.gpg --import /tmp/intel-key.pub && \
    rm /tmp/intel-key.pub && \
    echo "deb [signed-by=/etc/apt/trusted.gpg.d/intel.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y intel-oneapi-mkl-devel && \
    rm -rf /var/lib/apt/lists/* && \
    echo "source /opt/intel/oneapi/setvars.sh" >> /etc/bash.bashrc

# Add system packages
RUN apt-get update && \
    apt-get -y install \
    libc++-dev \
    libc++abi-dev \
    && rm -rf /var/lib/apt/lists/*
