# Construct Tensor Program (Abstraction)
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(A: T.Buffer(128, "float32"),
             B: T.Buffer(128, "float32"),
             C: T.Buffer(128, "float32")):
        # extra annotations for the function
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in range(128):
            with T.block("C"):
                # delcare a data parallel iterator on spatial domain
                vi = T.axis.spatial(128, i)
                C[vi] = A[vi] + B[vi]

# TVMScrips is a way for us to express tensor program in **python ast**
# Note that this code do not actually correspond to a python program, but tensor program that can be used in MLC process.
# The language is designed to align with python syntax with additional structures to facilitate analysis and transformation.

print(type(MyModule))
# tvm.ir.module.IRModule
# IR Module data structure, which is used to **hold a collection of tensor functions**
# show() function to get a highlighted string based representation of the IRModule 
# This function is quite useful for inspecting the moduel during each step of transformation.
print(MyModule.show())

# Build & Run
rt_mod = tvm.build(MyModule, target = "llvm") # The runtime module for CPU backends
print(type(rt_mod))

# after build, mod contains a collection of runnable functions. we can retrieve each function by its name.
func = rt_mod["main"]
print(func)

a = tvm.nd.array(np.arange(128, dtype="float32"))
b = tvm.nd.array(np.ones(128, dtype="float32"))
c = tvm.nd.empty((128,), dtype="float32")

print(a, b, c)
func(a, b, c)
print(a, b, c)

# Transform the Tensor Program
# Tensor program can be transformed using an auxilary data structure called schedule
sch = tvm.tir.Schedule(MyModule)
print(type(sch))

# 1. split loop

# get block by its name
block_c = sch.get_block("C")

# Get loops surrounding the block
(i,) = sch.get_loops(block_c)

# Tile the loop nesting.
i0, i1, i2 = sch.split(i, factors=[None, 4, 4])
sch.mod.show()

sch.reorder(i0, i2, i1)
sch.mod.show()

sch.parallel(i0)
print(sch.mod.show())

transformed_mod = tvm.build(sch.mod, target = "llvm")
transformed_mod["main"](a, b, c)

# Constructing Tensor Program using Tensor Expressions
# previous: directly use TVMScript to construct the tensor program <- helpful to constrcut these functions pragmatically from existing definitions
# Tensor expression is an API that helps us to build some of the expression-like array computations

from tvm import te

# declare the computation using the expression API
A = te.placeholder((128,), name = "A")
B = te.placeholder((128,), name = "B")
C = te.compute((128,), lambda i: A[i] + B[i], name = "C")

# create a function with the specified list of arguments.
func = te.create_prim_func([A, B, C])

# mark the function name is main
func = func.with_attr("global_symbol", "main")
ir_mod_from_te = IRModule({"main": func})

ir_mod_from_te.show()

### Transforming a matrix multiplication program using tensor expression

from tvm import te

M = 1024
K = 1024
N = 1024

# The default tensor type in tvm
dtype = "float32"

target = "llvm"
dev = tvm.device(target, 0)

# Algorithm
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name = "A")
B = te.placeholder((K, N), name = "B")
C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis = k), name = "C")

# Default Schedule
func = te.create_prim_func([A, B, C])
func = func.with_attr("global_symbol", "main")
ir_module = IRModule({"main" : func})
print(ir_module.show())

func = tvm.build(ir_module, target="llvm") # The module for CPU backends.

a = tvm.nd.array(np.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(np.random.rand(K, N).astype(dtype), dev)
c = tvm.nd.array(np.zeros((M, N), dtype = dtype), dev)

func(a, b, c)

evaluator = func.time_evaluator(func.entry_name, dev, number = 1)
print("Baseline: %f" % evaluator(a, b, c).mean)

### Transform the loop access pattern to make it more cache friendly 
sch = tvm.tir.Schedule(ir_module)
print(type(sch))
block_c = sch.get_block("C")

(y, x, k) = sch.get_loops(block_c)

block_size = 32
yo, yi = sch.split(y, [None, block_size])
xo, xi = sch.split(x, [None, block_size])

sch.reorder(yo, xo, k, yi, xi)
sch.mod.show()

func = tvm.build(sch.mod, target = "llvm")

c = tvm.nd.array(np.zeros((M, N), dtype = dtype), dev)
func(a, b, c)

evaluator = func.time_evaluator(func.entry_name, dev, number = 1)
print("after transformation: %f" % evaluator(a, b, c).mean)

# 정리: Primitive Tensor Function == abstraction)을 IRModule로 만들어 build하여 실제로 사용할 수 있음 
# 다양한 Abstraction 중 최적의 (efficiency) abstraction으로 transformation하는 것을 목표로 함 
# 1. 이를 위해서 직접 IRModule의 script로 Tensor Program을 만들어서 target에 맞게 build하여 사용 
# 2. Tensor Expression으로 Tensor의 연산 식을 정의하여 IRModule로 변환시켜build할 수 있음 
# 3. 여러 Transformation을 위해 tensor program을 auxilary data structure인schedule로 변환 
# 4. schedule의 정해진 block의 loop를 변환시켜 loop splitting , reorder, ... 등 여러 transformation 적용 가능 



