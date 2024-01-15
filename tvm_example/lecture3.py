
# Machine Learning Compilation의 대부분의 과정은 Tensor function들 간의 transformation으로 볼 수 있음 
# 문제 1: Tensor function을 표현하기 위한 가능한 abstraction은 무엇인가?
# 문제 2: Tensor function들 사이의 가능한 transformation들은 무엇인가?

# 우리는 primitive tensor function에 초점을 맞춰서 일부를 다룬다.

# 1) Learning One Tensor Program Abstraction - Tensor IR

# 지금까지 primitive tensor function에 대해 공부하였으며, tensor program abstraction의 상위 개념에 대해 살펴보았다.
# 이제 Tensor IR이라는 tensor program의 추상화의 특정 instance를 배울 것이다. Tensor IR은 standard machine learning compilation framework 중 하나인 
# TVM의 tensor program abstraction이다.

import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

# Tensor Program Abstraction의 주요한 목적은 loop 및 스레딩, specialized hardware instruction의 사용 및 메모리 엑세스와 같은 대응하는 하드웨에 대한 acceleration choice를 
# represent하는 것이다.

# Motivation example: Y = matmul(A, B), C = ReLU(Y) = max(Y, 0)
# 1. Numpy 

dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)
c_mm_relu = np.maximum(a_np @ b_np, 0)

# Numpy는 이러한 연산을 OpenBLAS와 같은 라이브러리와 low-level C언어로 구성된 일부를 호출하여 이러한 계산을 실행한다.
# 해당 계산을 implementation하기 위한 가능한 방법은 무엇인가?

# 이를 설명하기 위해 Numpy API의 제한된 subset으로 구현된 detail을 살펴보자 (low-level numpy)

# 우리는 loop computation을 보여주기 위해 array function 대신 loop를 사용할 것이다. 
# 또한, 가능하면 항상 numpy.empty를 사용하여 배열을 명시적으로 할당하고 전달한다.

# 이는 일반적으로 numpy program을 작성하는 방법이 아님을 명심하고, under the hood에서 발생하는 일과 유사하다는 알아라.
# 대부분의 실제 deployment에서는 computation과 allocation을 분리하여 다룬다.
# 특정 라이브러리들은 다른 형태의 loop와 산술 연산을 사용하여 계산을 수행한다.
# 물론 이들은 주로 C언어와 같은 하위 언어를 사용하여 구현된다.

def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype = "float32")
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)

# 위의 program은 mm_relu의 하나의 구현을 나타낸다. 
# 위의 implementation은 2개의 part로 구성되어있다. 
# 1) intermediate storage Y를 allocate하며 matmul의 결과를 store한다.
# 2) 2번째 sequence loop를 통해 relu를 계산한다. 
# 이것이 mm_relu를 구현하는 유일한 방법이 아니라는 것에 유의해라.
# 그럼에도 불구하고, 이것은 mm_relu를 구현하는 하나의 방법이며
# 우리는 array 계산을 통해 우리의 결과를 원래의 것과 비교하여 코드의 정확성을 확인할 수 있다.
# 우리는 돌아와서 튜토리얼의 후반부에 다른 가능한 방법들을 살펴볼 것이다.

c_np = np.empty((128, 128), dtype = dtype)
lnumpy_mm_relu(a_np, b_np, c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol = 1e-5)

# 위의 코드는 mm_relu의 under the hood implementation을 어떻게 수행할 수 있는지 보여준다. 
# 물론 코드는 python interpreter이기 때문에 훨씬 느리게 수행될 것이다.
# 그럼에도 불구하고, 예제 numpy 코드는 그러한 계산들이 실제 구현에서 사용되는 가능한 모든 요소들을 포함하고 있다.

# 1) Multi-dimensional buffer (arrays)
# 2) Loops over array dimensions.
# 3) Computations statements are executed under the loops

# low-level numpy example을 마음에 새기고, 이제 우리는 Tensor IR에 대해서 소개한다.
# 아래의 코드는 mm_relu의 Tensor IR implementation code block을 보여준다.
# 특정한 코드는 TVMScript라고 불리는 언어로 구현되어 있으며, 이는 python AST에 embedded되어있는 
# domain-specific dialect 중 하나이다.

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"),
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((128, 128), dtype = "float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


# 먼저, numpy와 TensorIR 사이의 직접적인 관련성이 있는 요소들을 검토하는 것으로 시작하자.
# 그러고 나서 우리는 다시 돌아와서 numpy프로그램에 속하지 않는 추가적인 요소들을 검토한다.


# 1. Function Parameters and Buffers
# 먼저 function의 parameter들을 봐보자, function parameter들은 numpy function과 같은 set의 parameter들이다.
'''
    # TensorIR
    def mm_relu(A: T.Buffer[(128, 128), "float32"],
		B: T.Buffer[(128, 128), "float32"],
		C: T.Buffer[(128, 128), "float32"]):
	...
    # numpy
    def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
	...
'''
# 여기서 A, B, C는 T.buffer의 type이며, shape argument는 (128, 128)이고, type은 float32이다.
# 이러한 추가적인 정보는 가능한 MLC process가 shape와 dtype에 특화된 코드를 생성하는 것을 돕는다.
# 마찬가지로 TensorIR은 또한 intermediate result를 allocation을 위한 buffer type 또한 사용한다.


'''
    # TensorIR
    Y = T.alloc_buffer((128, 128), dtype="float32")
    # numpy
    Y = np.empty((128, 128), dtype="float32")
'''

# 2. For Loop Iterations
# loop iteration과 direct correspondence하는 것이 있다. T.grid는 Tensor IR에서 nested iterator를 
# 작성할 수 있게 해주는 syntactic sugar이다.

'''
    # TensorIR
    for i, j, k in T.grid(128, 128, 128):

    # numpy
    for i in range(128):
	for j in range(128):
	    for k in range(128):
'''


# 3. Computational Block
# Numpy 구현과 TensorIR의 구현에서 주요한 다른 점은 computational statement에서 나온다.
# Tensor IR은 T.block이라 불리는 추가적인 construct를 가지고 있다.

'''
    # TensorIR
    with T.block("Y"):
	vi = T.axis.spatial(128, i)
	vj = T.axis.spatial(128, j)
	vk = T.axis.reduce(128, k)
	with T.init():
	    Y[vi, vj] = T.float32(0)
	Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

    # coressponding numpy code
    vi, vj, vk = i, j, k
    if vk == 0:
	Y[vi, vj] = 0
    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
'''

# block은 TensorIR의 computation에서 basic unit이다. 
# block은Numpy의 plain code와는 달리 몇가지의 추가적인 information을 가지고 있다.
# block은 block axes의 set (vi, vj, vk)를 가지고 있으며, computation은 이 set에 의해서 정의된다.

'''
    vi = T.axis.spatial(128, i)
    vj = T.axis.spatial(128, j)
    vk = T.axis.reduce(128, k)
'''

# 위의 3 line들은 block axis에 대한 핵심 속성을 다음과 같은 구문으로 선언한다.

'''
    [block_axis] = T.axis.[axis_type]([axis_range], [mapped_value])
'''

# 이 3줄에는 다음과 같은 정보가 포함되어있다.
# 1. They define where should vi, vj, vk be bound to (in this case i, j k).
# 2. They declare the original range that the vi, vj, vk are supposed to be (the 128 in T.axis.spatial(128, i))
# 3. They declare the properties of the iterators (spatial, reduce)
# 이러한 속성을 하나씩 살펴보면, 
# 1. vi = T.axis.spatial(128, i)는 사실상 vi = i를 의미한다.
# 2. [axis_range]의 값은 [block_axis]의 예상 범위를 제공한다. vi가 (0, 128)의 범위

# 3. Block Axis Properties
# block axis property에 대해서 더 자세히 살펴보자 
# 이러한 axis들은 수행 중인 연산에 대한 축의 관계를 표시한다.
# https://mlc.ai/chapter_tensor_program/case_study.html
# 아래 그림은 블록(iteration) 축과 블록 Y의 read write 작업의 관계를 요약한 것이다.
# 블록이 Y에 대한 업데이트(reduction)하는 것은 엄밀하게 말하면, 다른 블록으로부터 Y의 값이 필요 없기(read) 때문에 
# write로 표시한다.


