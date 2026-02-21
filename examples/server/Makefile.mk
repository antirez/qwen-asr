# examples/server/Makefile.mk — included by the root Makefile via wildcard
# Builds qwen_asr_server (C++17) against the BLAS-accelerated C objects.
# Run: make server

.PHONY: server

SERVER_TARGET   = qwen_asr_server
SERVER_CXX      = g++
SERVER_CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -ffast-math \
                  -I. -Iexamples/server

ifeq ($(UNAME_S),Darwin)
server: SERVER_CFLAGS_EX  = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
server: SERVER_LDFLAGS_EX = -framework Accelerate -lm -lpthread
else
server: SERVER_CFLAGS_EX  = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
server: SERVER_LDFLAGS_EX = -lopenblas -lm -lpthread
endif
server:
	@$(MAKE) $(OBJS) CFLAGS="$(SERVER_CFLAGS_EX)"
	$(SERVER_CXX) $(SERVER_CXXFLAGS) $(SERVER_CFLAGS_EX) \
	    examples/server/server.cpp $(OBJS) \
	    $(SERVER_LDFLAGS_EX) -o $(SERVER_TARGET)
	@echo ""
	@echo "Built $(SERVER_TARGET)"

clean::
	rm -f $(SERVER_TARGET)
