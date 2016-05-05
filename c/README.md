JDA-C
=====

Optimized C Version of JDA Detection Part.

There's something you should pay attention to.

- The code is optimized for VC12, no guarantee for GCC or other compilers, but should run much faster than the cpp code which is not optimized.
- The detection part is different from the default detection function `detectMultiScale` in `cascador.cpp`, it is more like the function `detectMultiScale1`.
- The code may give different results with different parameters. The detection speed can also be very different.
