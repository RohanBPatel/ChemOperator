FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    git \
    make \
    pkg-config \
    python3 \
    python3-pip \
    libboost-dev \
    && add-apt-repository -y ppa:cantera-team/cantera \
    && apt-get update \
    && apt-get install -y python3-cantera libcantera-dev cantera-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
COPY external/Quasi-2D-packbed /opt/Quasi-2D-packbed

WORKDIR /opt/Quasi-2D-packbed/compile

RUN cat > Makefile <<'EOF'
.RECIPEPREFIX := >
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall
SRC = main.cpp BVP_solver.cpp canteraInput.cpp
TARGET = packbedQ2D
CXXFLAGS += $(shell pkg-config --cflags cantera || true)
LDLIBS += $(shell pkg-config --libs cantera || echo -lcantera)

all: $(TARGET)

$(TARGET): $(SRC)
>$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDLIBS)

clean:
>rm -f $(TARGET) *.o
EOF

RUN make -j$(nproc)

WORKDIR /work
ENTRYPOINT ["/opt/Quasi-2D-packbed/compile/packbedQ2D"]
