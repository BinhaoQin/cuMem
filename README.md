# cuMem

CUDA Memory Management Wrapper with Type Safety.

## 1. API Descriptions

- `cuConstant.h` mainly works with `__constant__` memory
- `cuShared.h` is dangerous: **strips off type info to avoid constructor**
- `cuManaged.h` is mainly for interfacing GPU and CPU code
