# examples/bench/Makefile.mk — included by the root Makefile via wildcard
# Builds qwen_asr_bench (pure C) against the BLAS-accelerated C objects.
# Run: make bench

.PHONY: bench

BENCH_TARGET = qwen_asr_bench

ifeq ($(UNAME_S),Darwin)
bench: BENCH_CFLAGS  = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
bench: BENCH_LDFLAGS = -framework Accelerate -lm -lpthread
else
bench: BENCH_CFLAGS  = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
bench: BENCH_LDFLAGS = -lopenblas -lm -lpthread
endif
bench:
	@$(MAKE) $(OBJS) CFLAGS="$(BENCH_CFLAGS)"
	$(CC) $(BENCH_CFLAGS) -I. \
	    examples/bench/bench.c $(OBJS) \
	    $(BENCH_LDFLAGS) -o $(BENCH_TARGET)
	@echo ""
	@echo "Built $(BENCH_TARGET)"

clean::
	rm -f $(BENCH_TARGET)
