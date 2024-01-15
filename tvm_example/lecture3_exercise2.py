import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

# Exercise 1: Element-wise Addition

a = np.arange(16).reshape(4, 4)
b = np.arange(16, 0, -1).reshape(4, 4)

# numpy version
c_np = a + b
print(c_np)

# low_level version
def lnumpy_add(a: np.ndarray, b:np.ndarray, c:np.ndarray):
    for i in range(4):
        for j in range(4):
            c[i, j] = a[i, j] + b[i, j]

c_lnumpy = np.empty((4, 4), dtype = np.int64)
lnumpy_add(a, b, c_lnumpy)
print(c_lnumpy)

# TensorIR Version
@tvm.script.ir_module
class MyAdd:
    @T.prim_func
    def add(A: T.Buffer((4, 4), "int64"),
            B: T.Buffer((4, 4), "int64"),
            C: T.Buffer((4, 4), "int64")):
        T.func_attr({"global_symbol": "add"})
        for i, j in T.grid(4, 4):
            with T.block("C"):
                vi = T.axis.spatial(4, i)
                vj = T.axis.spatial(4, j)
                C[vi, vj] = A[vi, vj] + B[vi, vj]
rt_lib = tvm.build(MyAdd, target = "llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty((4, 4), dtype = np.int64))

rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)
print(c_tvm)

# Exercise 2. Broadcast Add
a = np.arange(16).reshape(4, 4)
b = np.arange(4, 0, -1).reshape(4)

c_np = a + b
print(c_np)

@tvm.script.ir_module
class MyBroadcastAdd:
    @T.prim_func
    def add(A: T.Buffer((4, 4), "int64"),
            B: T.Buffer((4), "int64"),
            C: T.Buffer((4, 4), "int64")):
        T.func_attr({"global_symbol" : "broadcast_add", "tir_noalias": True})

        for i, j in T.grid(4, 4):
            with T.block("C"):
                vi = T.axis.spatial(4, i)
                vj = T.axis.spatial(4, j)

                C[vi, vj] = A[vi, vj] + B[vj]

rt_lib = tvm.build(MyBroadcastAdd, target = "llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty((4, 4), dtype = np.int64))

rt_lib["broadcast_add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)
print(c_tvm)


# Exercise 3. 2D Convolution
# Stride = 1, padding = 0, NCHW layout
N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)

# torch version
import torch

data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
conv_torch = conv_torch.numpy().astype(np.int64)
print(conv_torch)

@tvm.script.ir_module
class MyConv:
    @T.prim_func
    def conv(data: T.Buffer((1, 1, 8, 8), "int64"),
            weight: T.Buffer((2, 1, 3, 3), "int64"),
            out: T.Buffer((1, 2, 6, 6), "int64")):
        T.func_attr({"global_symbol" : "conv", "tir_noalias": True})
        for B, CO, I, J, CI, KH, KW in T.grid(1, 2, 6, 6, 1, 3, 3):
            with T.block("out"):
                vb = T.axis.spatial(1, B)
                vco = T.axis.spatial(2, CO)
                vi = T.axis.spatial(6, I)
                vj = T.axis.spatial(6, J)
                vci = T.axis.reduce(1, CI)
                kh = T.axis.reduce(3, KH)
                kw = T.axis.reduce(3, KW)
                with T.init():
                    out[vb, vco, vi, vj] = T.int64(0)
                out[vb, vco, vi, vj] = out[vb, vco, vi, vj] + data[vb, vci, vi + kh, vj + kw] * weight[vco, vci, kh, kw]

rt_lib = tvm.build(MyConv, target = "llvm")
a_tvm = tvm.nd.array(data)
b_tvm = tvm.nd.array(weight)
c_tvm = tvm.nd.array(np.empty((N, CO, OUT_H, OUT_W), dtype = np.int64))

rt_lib["conv"](a_tvm, b_tvm, c_tvm)
print(c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), conv_torch, rtol=1e-5)

# parallel vectorize Unroll
@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A: T.Buffer((4, 4), "int64"),
          B: T.Buffer((4, 4), "int64"),
          C: T.Buffer((4, 4), "int64")):
    T.func_attr({"global_symbol": "add"})
    for i, j in T.grid(4, 4):
      with T.block("C"):
        vi = T.axis.spatial(4, i)
        vj = T.axis.spatial(4, j)
        C[vi, vj] = A[vi, vj] + B[vi, vj]

sch = tvm.tir.Schedule(MyAdd)
block = sch.get_block("C", func_name="add")
i, j = sch.get_loops(block)
i0, i1 = sch.split(i, factors=[2, 2])
sch.parallel(i0)
sch.unroll(i1)
sch.vectorize(j)
print(sch.mod.script())

# Exercise 4. batched matmul relu
# hint
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((16, 128, 128), dtype="float32")
    for n in range(16):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    if k == 0:
                        Y[n, i, j] = 0
                    Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
    for n in range(16):
        for i in range(128):
            for j in range(128):
                C[n, i, j] = max(Y[n, i, j], 0)

# target program
@tvm.script.ir_module
class TargetModule:
    @T.prim_func
    def bmm_relu(A: T.Buffer((16, 128, 128), "float32"), B: T.Buffer((16, 128, 128), "float32"), C: T.Buffer((16, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
        Y = T.alloc_buffer([16, 128, 128], dtype="float32")
        for i0 in T.parallel(16):
            for i1, i2_0 in T.grid(128, 16):
                for ax0_init in T.vectorized(8):
                    with T.block("Y_init"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + ax0_init)
                        Y[n, i, j] = T.float32(0)
                for ax1_0 in T.serial(32):
                    for ax1_1 in T.unroll(4):
                        for ax0 in T.serial(8):
                            with T.block("Y_update"):
                                n, i = T.axis.remap("SS", [i0, i1])
                                j = T.axis.spatial(128, i2_0 * 8 + ax0)
                                k = T.axis.reduce(128, ax1_0 * 4 + ax1_1)
                                Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
                for i2_1 in T.vectorized(8):
                    with T.block("C"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + i2_1)
                        C[n, i, j] = T.max(Y[n, i, j], T.float32(0))

@tvm.script.ir_module
class MyBmmRelu:
  @T.prim_func
  def bmm_relu(A: T.Buffer((16, 128, 128), "float32"),
          B: T.Buffer((16, 128, 128), "float32"),
          C: T.Buffer((16, 128, 128), "float32")):
    T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
    Y = T.alloc_buffer([16, 128, 128], dtype = "float32")
    for i0, i1, i2, i3 in T.grid(16, 128, 128, 128):
      with T.block("Y"):
        n = T.axis.spatial(16, i0)
        i = T.axis.spatial(128, i1)
        j = T.axis.spatial(128, i2)
        k = T.axis.reduce(128, i3)
        with T.init():
            Y[n, i, j] = T.float32(0)
        Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
    
    for i0, i1, i2 in T.grid(16, 128, 128):
        with T.block("C"):
            n = T.axis.spatial(16, i0)
            i = T.axis.spatial(128, i1)
            j = T.axis.spatial(128, i2)

            C[n, i, j] = T.max(Y[n, i, j], T.float32(0))


sch = tvm.tir.Schedule(MyBmmRelu)
# TODO: transformations
# Hints: you can use
# `IPython.display.Code(sch.mod.script(), language="python")`
# or `print(sch.mod.script())`
# to show the current program at any time during the transformation.

# Step 1. Get blocks
Y = sch.get_block("Y", func_name="bmm_relu")
C = sch.get_block("C", func_name="bmm_relu")
# Step 2. Get loops
b, i, j, k = sch.get_loops(Y)

# Step 3. Organize the loops
k0, k1 = sch.split(k, factors = [None, 4])
j0, j1 = sch.split(j, factors = [None, 8])
sch.reorder(b, i, j0, k0, k1, j1)
sch.reverse_compute_at(C, j0)

# Step 4. decompose reduction
sch.parallel(b)
Y_init = sch.decompose_reduction(Y, k0)
sch.vectorize(sch.get_loops(Y_init)[-1])
sch.unroll(k1)
sch.vectorize(sch.get_loops(C)[-1])

print(sch.mod.script())

tvm.ir.assert_structural_equal(sch.mod, TargetModule)
print("Pass")
