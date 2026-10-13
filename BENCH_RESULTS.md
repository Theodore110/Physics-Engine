# physics2d benchmark
SIMD backend : x86_64 SSE2 intrinsics
profile      : release (opt 3, fat LTO), single thread, dt = 1/60, 8 vel iters, 3 pos iters, 7 repeats/window

## impact (steps 20-320)
| bodies | pairs/step | contacts/step | median step | best step | FPS (median) | ns/body/step | allocs in window |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 561 | 483 | 106.1 µs | 97.2 µs | 9427 | 424 | 0 |
| 500 | 1103 | 966 | 222.8 µs | 220.2 µs | 4488 | 446 | 0 |
| 1000 | 2228 | 1928 | 460.3 µs | 450.9 µs | 2172 | 460 | 0 |
| 2000 | 4383 | 3773 | 980.4 µs | 970.1 µs | 1020 | 490 | 0 |
| 4000 | 8496 | 7294 | 2163.6 µs | 2037.3 µs | 462 | 541 | 0 |

## settled (steps 900-1200)
| bodies | pairs/step | contacts/step | median step | best step | FPS (median) | ns/body/step | allocs in window |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 597 | 528 | 95.4 µs | 93.8 µs | 10484 | 382 | 0 |
| 500 | 1189 | 1071 | 229.5 µs | 226.5 µs | 4358 | 459 | 0 |
| 1000 | 2476 | 2184 | 531.3 µs | 505.0 µs | 1882 | 531 | 0 |
| 2000 | 4980 | 4418 | 1082.3 µs | 1072.0 µs | 924 | 541 | 0 |
| 4000 | 10101 | 8987 | 2838.7 µs | 2304.5 µs | 352 | 710 | 0 |

## per-phase breakdown, settled window, µs/step (probes on)
| bodies | integrate | broad | narrow | solve (vel+pos) | total |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.3 | 23.1 | 6.4 | 65.5 | 95.4 |
| 500 | 1.2 | 121.5 | 23.1 | 207.3 | 353.1 |
| 1000 | 1.2 | 160.9 | 28.0 | 302.0 | 492.2 |
| 2000 | 3.3 | 421.2 | 66.0 | 725.4 | 1215.9 |
| 4000 | 6.2 | 818.1 | 139.1 | 1546.7 | 2510.1 |

## per-phase breakdown, impact window, µs/step (probes on)
| bodies | integrate | broad | narrow | solve (vel+pos) | total |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.3 | 29.4 | 6.1 | 61.9 | 97.8 |
| 500 | 0.6 | 68.2 | 12.0 | 133.2 | 214.0 |
| 1000 | 1.2 | 155.6 | 25.6 | 264.3 | 446.7 |
| 2000 | 2.3 | 336.8 | 51.5 | 552.2 | 942.9 |
| 4000 | 5.7 | 737.3 | 112.3 | 1187.1 | 2042.4 |

## explicit SIMD vs scalar reference (N = 1000 and 4000, settled window, µs/step, best of 7)
| bodies | kernel | scalar | SIMD | speed-up |
|---:|---|---:|---:|---:|
| 1000 | integrate (vel + pos) | 1.24 | 1.26 | 0.98x |
| 1000 | broad phase (AABB+grid+pairs) | 169.65 | 166.11 | 1.02x |
| 4000 | integrate (vel + pos) | 6.06 | 5.87 | 1.03x |
| 4000 | broad phase (AABB+grid+pairs) | 803.12 | 768.05 | 1.05x |

## determinism (N = 1000, 1200 steps)
run 1 (SIMD)   : 10439adf8664a907
run 2 (SIMD)   : 10439adf8664a907   identical to run 1: true
run 3 (scalar) : 10439adf8664a907   identical to SIMD  : true
(compare this hash across builds/platforms, e.g. `--features portable_simd`)
