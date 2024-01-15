#In the lecture, we use a tmm intrinsic to demonstrate the Tensorize progress. In this tutorial, we will integrate TensorIR to Tensor Cores on NVIDIA GPUs. Note that Tensor Cores are only supported on GPUs with with Volta and newer architectures (e.g. V100, T4, RTX-20X0, A100, RTX-30X0). Unfortunately, most GPUs provided by Colab are too old to support Tensor Cores. You may have to prepare your own devices for this tutorial.

# What are Tensor Cores?
# Tensor Cores are programmable matrix-multiply-and-accumulate units on GPUs with Volta and newer architectures. Each Tensor Core provides a matrix processing array which performs the operation D = A * B + C, where A, B, C and D are 16x16 matrices if we use nvcuda::wmma in CUDA language. The matrix multiply inputs A and B are fp16 matrices, while the accumulation matrices C and D may be fp16 or fp32 matrices.

# CUDA programmers can only use "warp-level" primitive wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag) to perform 16x16x16 half-precision matrix multiplication on tensor cores. Before invoking the matrix multiplication, programmers must load data from memory into registers with primitive wmma::load_matrix_sync, explicitly (similar to what we do in the tmm demo of Chapter 6 part 2). The NVCC compiler translates that primitive into multiple memory load instructions. At run time, every thread loads 16 elements from matrix A and 16 elements from B.

import tvm
from tvm.script import tir as T
from tvm import tir

import numpy as np

@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(
            X: T.Buffer((1024, 1024), "float16"),
            Y: T.Buffer((1024, 1024), "float16"),
            Z: T.Buffer((1024, 1024), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Z[vi, vj] = T.float32(0)
                Z[vi, vj] += T.cast(X[vi, vk], "float32") * T.cast(Y[vj, vk], "float32")
# Note the computational body Z[vi, vj] += T.cast(X[vi, vk], "float32") * T.cast(Y[vj, vk], "float32") is a bit different. Since Tensor Cores loads data of fp16 but do calculation at fp32. So we must cast the data to fp32 before the fma operation.


# Memory Scope
# In traditional GPU schedule, we have global, shared and local memory scope. To support Tensor Cores, we add another three special memory scope: wmma.matrix_a, wmma.matrix_b and wmma.accumulator (which is similar to global.A_reg, global.B_reg and global.accumulator in the demo of Chapter 6 part 2). On hardware, all wmma scope stores at the on-chip registers level, the same place with local memory.

# Register Tensor Intrinsic
# Here we register all tensor core intrinsics, including load_matrix_a, load_matrix_b, wmma_fill (init C = 0), wmma_sync (do accumulation calculation C += A * B) and store_matrix. In this tutorial, we won't explain how to write intrinsic but focus on how to apply given intrinsic to tensorized programs.

@T.prim_func
def wmma_load_a_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor = 16, scope= "shared")
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor = 16, scope= "wmma.matrix_a")

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vij = T.axis.remap("SS", [i, j])
                C[vii, vij] = A[vii, vij]
@T.prim_func
def wmma_load_a_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")

    A = T.match_buffer(
            a,
            (16, 16),
            "float16",
            align = 128,
            offset_factor = 16,
            scope = "shared",
            strides = [s1, s0],
    )
    C = T.match_buffer(c, (16, 16), "float16", align = 128, offset_factor = 16, scope = "wmma.matrix_a")
    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_load_matrix_sync(
                C.data,
                16,
                16,
                16,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                A.access_ptr("r"),
                s1,
                "row_major",
                dtype = "handle",
            )
        )

@T.prim_func
def wmma_load_b_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align = 128, offset_factor=16, scope = "shared")
    C = T.match_buffer(c, (16, 16), "float16", align = 128, offset_factor=16, scope = "wmma.matrix_b")
    
    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]

@T.prim_func
def wmma_load_b_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(
            a,
            (16, 16),
            "float16",
            align = 128,
            offset_factor=16,
            scope = "shared",
            strides = [s1, s0],
    )
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor = 16, scope = "wmma.matrix_b")

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_load_matrix_sync(
                C.data,
                16,
                16,
                16,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                A.access_ptr("r"),
                s1,
                "col_major",
                dtype = "handle",
            )
        )

@T.prim_func
def wmma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align = 128, offset_factor = 16, scope = "wmma.matrix_a")
    B = T.match_buffer(b, (16, 16), "float16", align = 128, offset_factor = 16, scope = "wmma.matrix_b")
    C = T.match_buffer(c, (16, 16), "float32", align = 128, offset_factor = 16, scope = "wmma.accumulator")

    with T.block("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block(""):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] += T.cast(A[vii, vkk], "float32") * T.cast(B[vjj, vkk], "float32")
                
@T.prim_func
def wmma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a")
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b")
    C = T.match_buffer(
        c, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )

    with T.block("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_mma_sync(
                C.data,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                A.data,
                A.elem_offset // 256 + T.floordiv(T.floormod(A.elem_offset, 256), 16),
                B.data,
                B.elem_offset // 256 + T.floordiv(T.floormod(B.elem_offset, 256), 16),
                C.data,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                dtype="handle",
            )
        )

@T.prim_func
def wmma_fill_desc(c: T.handle) -> None:
    C = T.match_buffer(
        c, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )

    with T.block("root"):
        T.reads()
        T.writes(C[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = T.float32(0)

@T.prim_func
def wmma_fill_impl(c: T.handle) -> None:
    C = T.match_buffer(
        c, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        T.reads()
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_fill_fragment(
                C.data,
                16,
                16,
                16,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                T.float32(0),
                dtype="handle",
            )
        )

@T.prim_func
def wmma_store_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="global")
    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]

@T.prim_func
def wmma_store_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c,
        (16, 16),
        "float32",
        align=128,
        offset_factor=16,
        scope="global",
        strides=[s1, s0],
    )
    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_store_matrix_sync(
                A.data,
                16,
                16,
                16,
                A.elem_offset // 256 + T.floordiv(T.floormod(A.elem_offset, 256), 16),
                C.access_ptr("w"),
                s1,
                "row_major",
                dtype="handle",
            )
        )
try:
    # handle exception if we register multi times
    tir.TensorIntrin.register("wmma_load_a", wmma_load_a_desc, wmma_load_a_impl)
    tir.TensorIntrin.register("wmma_load_b", wmma_load_b_desc, wmma_load_b_impl)
    tir.TensorIntrin.register("wmma_sync", wmma_sync_desc, wmma_sync_impl)
    tir.TensorIntrin.register("wmma_fill", wmma_fill_desc, wmma_fill_impl)
    tir.TensorIntrin.register("wmma_store", wmma_store_desc, wmma_store_impl)
except ValueError:
    pass

# Blockize the Tensorized Computation
# As says in the lecture, we can use TensorIR to represent a group of tensorized computation with a Block. We can directly write a TensorIR program with Block, but also can tile the loops and blockize it. Remember that a wmma operation works on a 16x16x16 matrix multiplication, we need to tile the loops while inner-most loops are 16x16x16.

sch = tir.Schedule(MatmulModule)
block = sch.get_block("matmul")
i, j, k = sch.get_loops(block)

i, ii = sch.split(i, factors= [None, 16])
j, ji = sch.split(j, factors= [None, 16])
k, ki = sch.split(k, factors= [None, 16])

sch.reorder(i, j, k, ii, ji, ki)
print(sch.mod.show())

wmma_sync = sch.blockize(ii)
print(sch.mod.show())

# Tile the loop nesting and bind threadIdx
# Warp-level Operation
# Note that all Tensor Core instructions are warp-level instructions, which means all 32 threads in a warp should do this instruction simultaneously. Making threadIdx.x extent=32 is one of the easiest way to solve this. Then We can bind threadIdx.x to any loops except those contain Tensor Core intrinsics directly or indirectly. Also note that it is not the unique solution. The only thing we should do is to make sure all threads in a warp can call Tensor Core at the same time.

i0, i1, i2 = sch.split(i, factors=[8, 4, 2])
j0, j1, j2 = sch.split(j, factors=[8, 4, 2])
k0, k1, k2 = sch.split(k, factors=[16, 2, 2])

sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2, k2)
bx = sch.fuse(i0, j0)
sch.bind(bx, "blockIdx.x")
ty = sch.fuse(i1, j1)
sch.bind(ty, "threadIdx.y")
# We can't bind to `threadIdx.x` since we have warp-level operators under the loop
print(sch.mod.show())

X_shared = sch.cache_read(wmma_sync, read_buffer_index=0, storage_scope="shared")
Y_shared = sch.cache_read(wmma_sync, read_buffer_index=1, storage_scope="shared")


def schedule_shared(block):
    sch.compute_at(block, k0)
    x, y = sch.get_loops(block)[-2:]
    fused = sch.fuse(x, y)
    x0, x1, x2, x3 = sch.split(fused, factors=[None, 16, 32, 8])
    sch.bind(x1, "threadIdx.y")
    # here we must bind threadIdx.x == 32 to satisfy the requirements of warp-level operation.
    sch.bind(x2, "threadIdx.x") 
    sch.vectorize(x3)


schedule_shared(X_shared)
schedule_shared(Y_shared)
print(sch.mod.show())

# Cache the input and output data to special memory scope
# Tensor Cores can not direct use data in shared memory or local memory. We must cache the data into wmma.matrix_a, wmma.matrix_b and update the computation in wmma.accumulator
X_local = sch.cache_read(wmma_sync, 0, storage_scope="wmma.matrix_a")
Y_local = sch.cache_read(wmma_sync, 1, storage_scope="wmma.matrix_b")
sch.compute_at(X_local, k1)
sch.compute_at(Y_local, k1)
print(sch.mod.show())

write_back_block = sch.cache_write(wmma_sync, 0, storage_scope="wmma.accumulator")
sch.reverse_compute_at(write_back_block, ty)
print(sch.mod.show())

# Tile the Tensor Core memory copying
# wmma.load_matrix and wmma.store_matrix perform memory copies with 16x16 matrices. Then we need to tile the copy blocks to match the intrinsic.
def schedule_copy(block):
    x, y = sch.get_loops(block)[-2:]
    x0, x1 = sch.split(x, factors=[None, 16])
    y0, y1 = sch.split(y, factors=[None, 16])
    sch.reorder(x0, y0, x1, y1)

schedule_copy(X_local)
schedule_copy(Y_local)
schedule_copy(write_back_block)
print(sch.mod.show())

# Tensorize
# Before tensorize, we decompose_reduction first, because the wmma_sync and wmma_fill are two intrinsics and need to tensorize twice for both init block and update block
init = sch.decompose_reduction(wmma_sync, k0)
print(sch.mod.show())

sch.tensorize(sch.get_loops(X_local)[-2], "wmma_load_a")
sch.tensorize(sch.get_loops(Y_local)[-2], "wmma_load_b")
sch.tensorize(init, "wmma_fill")
sch.tensorize(wmma_sync, "wmma_sync")
sch.tensorize(sch.get_loops(write_back_block)[-2], "wmma_store")
print(sch.mod.show())





