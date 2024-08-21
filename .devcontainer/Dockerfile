# Use the base image
FROM mcr.microsoft.com/devcontainers/base:noble

# Argument for LLVM version
ARG LLVM_VERSION=19
ARG CMAKE_VERSION=3.29.5

# C++ development environment, uses clang, libc++, and ninja
ENV CC=clang-$LLVM_VERSION
ENV CXX=clang++-$LLVM_VERSION
ENV FC=gfortran
ENV CMAKE_GENERATOR=Ninja
ENV CXXFLAGS=-stdlib=libc++

# Add the LLVM repository and install the specified version of LLVM
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    wget \
    gnupg \
    software-properties-common \
    && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-$LLVM_VERSION main" \
    && apt-get update \
    && apt-get -y install --no-install-recommends \
    clang-$LLVM_VERSION \
    clang-format-$LLVM_VERSION \
    clang-tidy-$LLVM_VERSION \
    clangd-$LLVM_VERSION \
    libc++-$LLVM_VERSION-dev \
    libc++abi-$LLVM_VERSION-dev \
    lldb-$LLVM_VERSION \
    lld-$LLVM_VERSION \
    ninja-build \
    gfortran \
    python3-numpy \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \ 
    && ln -s /usr/bin/clang-$LLVM_VERSION /usr/bin/clang \
    && ln -s /usr/bin/clang++-$LLVM_VERSION /usr/bin/clang++ \
    && ln -s /usr/bin/clang-format-$LLVM_VERSION /usr/bin/clang-format \
    && ln -s /usr/bin/clang-tidy-$LLVM_VERSION /usr/bin/clang-tidy \
    && ln -s /usr/bin/clangd-$LLVM_VERSION /usr/bin/clangd \
    && ln -s /usr/bin/lldb-$LLVM_VERSION /usr/bin/lldb \
    && ln -s /usr/bin/lld-$LLVM_VERSION /usr/bin/lld \
    && ln -f -s /usr/bin/ld.lld-$LLVM_VERSION /usr/bin/ld 

# download the key to system keyring and add signed entry to apt sources and configure the APT client to use Intel repository:
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

# install Intel OneAPI for MKL
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    intel-oneapi-mkl-devel

# Install specific version of CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-${CMAKE_VERSION} \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-${CMAKE_VERSION} \
    && ln -s /opt/cmake-${CMAKE_VERSION}/bin/* /usr/local/bin \
    && rm /tmp/cmake-install.sh

# Clean up
RUN apt-get -y remove wget gnupg software-properties-common
