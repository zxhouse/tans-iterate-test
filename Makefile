CC ?= gcc
CFLAGS ?= -O2 -march=native

all: src/libtans.so src/enum src/bench

src/libtans.so: src/tans.c
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $< -lm

src/enum: src/enum.c src/tans.c
	$(CC) $(CFLAGS) -o $@ src/enum.c src/tans.c -lm

src/bench: src/bench.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f src/libtans.so src/enum src/bench
