# Section 1: Model Preparation
import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
import torchvision
import tvm
import tvm.testing

from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from tvm import topi, relax, te
from tvm.script import tir as T

batch_size = 4
input_shape = (batch_size, 1, 28, 28)  # NCHW layout


def pytorch_model():
    list = []
    list.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), bias=True))
    list.append(nn.ReLU())
    list.append(nn.MaxPool2d(kernel_size=(2, 2)))
    list.append(nn.Flatten())
    list.append(nn.Linear(in_features=5408, out_features=100, bias=True))
    list.append(nn.ReLU())
    list.append(nn.Linear(in_features=100, out_features=10, bias=True))
    list.append(nn.Softmax(dim=1))
    #list.append(nn.LogSoftmax(dim=1))

    model = nn.Sequential(*list).cpu()
    name_map = {
        "0.weight": "conv2d_weight",
        "0.bias": "conv2d_bias",
        "4.weight": "linear0_weight",
        "4.bias": "linear0_bias",
        "6.weight": "linear1_weight",
        "6.bias": "linear1_bias",
    }
    for name, param in model.named_parameters():
        param.data = torch.from_numpy(weight_map[name_map[name]]).cpu()
    return model

# Load the weight map from file.
# The prediction accuracy of the weight map on test data is around 83.3%.
weight_map = pkl.load(open("fasionmnist_mlp_assignment_params.pkl", "rb"))
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        print_img = False
        for data, label in test_loader:
            data, label = data.cpu(), label.cpu()
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, label, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            if print_img:
                imshow(data[0])
                print("predict: {}, label: {}".format(class_names[pred[0][0]], class_names[label[0]]))
                print_img = False
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


test_data = torchvision.datasets.FashionMNIST(
    "./data",
    download=True,
    train=False,
    transform=transforms.Compose([transforms.ToTensor()])
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False)

test(pytorch_model(), test_loader)

# Section 2. Ingest Model From Pytorch

'''
To see the MLC abstraction of the end-to-end model, we need to ingest it from PyTorch and transform into TVMScript implementation. 
However, it is hard to manually do this. 
As you may have experienced in Exercise 1, writing a primitive tensor function for each model layer requires massive engineering efforts. 
Moreover, the manual writing process is error-prone - just imagine when you write dozens of lines of code while there exists some tiny bug in your implementation, finding the bug in could be annoying.

Fortunately, in TVM there is a much simpler way of doing this. 
TVM provides a utility relax.BlockBuilder that can construct end-to-end models step by step in an IRModule that starts empty. 
(Recall that in Lecture 4 we introduced the dataflow block design of Relax, our MLC abstraction on computational graph level. And here the "block" in "BlockBuilder" stands for the dataflow blocks in Relax functions.)

***Specifically, in BlockBuilder we have an emit_te API, that helps convert a Tensor Expression operator description, which was introduced in Lecture 3, into a call_tir operation to the operator's corresponding TensorIR function 
(call_tir was introduced in Lecture 4 as well.) 
Compared with manually writing TensorIR functions, writing their Tensor Expression description can be done within only a few lines of code, which reduces the amount of efforts and is less likely for us to make mistakes.***

The signature of emit_te is emit_te(func, *input), where func is a function that returns a Tensor Expression operator description, and *input is the inputs to func.

Let's start with an introducing example. 
In the code block below, relu is a function that returns a Tensor Expression description of a ReLU operator. 
To construct a Relax function that executes a single ReLU operator, in function emit_te_example we first define a BlockBuilder instance bb. We also define a 2-dimensional 128x128 tensor variable x, which will serve as the input tensor of the ReLU operation (as well as the input of the Relax function).

After that, we construct a Relax function main with x as input, using the with bb.function(name, [*input]) API. 
Then we construct a dataflow block. 
Inside the dataflow block, we first have a call_tir to a TensorIR implementation of ReLU operator, through emit_te. 
***The emit_te below generates a TensorIR function called "relu" in the IRModule, and add a call_tir(relu, (x,), (128, 128), dtype="float32") operation in the dataflow block. ***
And the call_tir is followed by a function return.

After this construction, the BlockBuilder bb contains the constructed IRModule, which can be got by bb.get().
'''

def relu(A):
    B = te.compute(shape=(128, 128), fcompute=lambda i, j: te.max(A[i, j], 0), name="B")
    return B

#def conv(X, W, B):
#    O = te.compute(shape=(batch_size, 32, 26, 26), fcompute=lambda i, j, 

def emit_te_example():
    bb = relax.BlockBuilder()
    # DynTensorType : (ndim, dtype, span)
    x = relax.Var("x", relax.TensorStructInfo((128, 128), "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit_te(relu, x)
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)
    return bb.finalize()

import IPython

mod = emit_te_example()
print(mod.script())

def create_model_via_emit_te2():
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo(input_shape, "float32"))

    conv2d_weight = relax.const(weight_map["conv2d_weight"], "float32")
    conv2d_bias = relax.const(weight_map["conv2d_bias"].reshape(1, 32, 1, 1), "float32")
    linear0_weight = relax.const(weight_map["linear0_weight"], "float32")
    linear0_bias = relax.const(weight_map["linear0_bias"].reshape(1, 100), "float32")
    linear1_weight = relax.const(weight_map["linear1_weight"], "float32")
    linear1_bias = relax.const(weight_map["linear1_bias"].reshape(1, 10), "float32")
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit_te(topi.nn.conv2d, x, filter = conv2d_weight, strides = 1, padding = 0, dilation = 1)
            lv1 = bb.emit_te(topi.nn.add, lv0, conv2d_bias)
            lv2 = bb.emit_te(topi.nn.relu, lv1)
            lv3 = bb.emit_te(topi.nn.pool2d, lv2, [2, 2], [2, 2], [1, 1], [0, 0, 0, 0], 'max')
            lv4 = bb.emit_te(topi.nn.flatten, lv3)
            lv5 = bb.emit_te(topi.nn.dense, lv4, linear0_weight, linear0_bias)
            lv6 = bb.emit_te(topi.nn.relu, lv5)
            lv7 = bb.emit_te(topi.nn.dense, lv6, linear1_weight, linear1_bias)
            #gv = bb.emit_te(topi.nn.dense, lv6, linear1_weight, linear1_bias)
            #gv = bb.emit_te(topi.nn.log_softmax, lv7, -1)
            gv = bb.emit_te(topi.nn.softmax, lv7, 1)
        bb.emit_func_output(gv)
    return bb.finalize()

def create_model_via_emit_te():
    bb = relax.BlockBuilder()
    #x = relax.Var("x", input_shape, relax.DynTensorType(batch_size, "float32"))
    x = relax.Var("x", relax.TensorStructInfo(input_shape, "float32"))

    conv2d_weight = relax.const(weight_map["conv2d_weight"], "float32")
    conv2d_bias = relax.const(weight_map["conv2d_bias"].reshape(1, 32, 1, 1), "float32")
    linear0_weight = relax.const(weight_map["linear0_weight"], "float32")
    linear0_bias = relax.const(weight_map["linear0_bias"].reshape(100,), "float32")
    linear1_weight = relax.const(weight_map["linear1_weight"], "float32")
    linear1_bias = relax.const(weight_map["linear1_bias"].reshape(10,), "float32")
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit_te(topi.nn.conv2d, x, filter = conv2d_weight, strides = 1, padding = 0, dilation = 1)
            lv1 = bb.emit_te(topi.nn.add, lv0, conv2d_bias)
            lv2 = bb.emit_te(topi.nn.relu, lv1)
            lv3 = bb.emit_te(topi.nn.pool2d, lv2, [2, 2], [2, 2], [1, 1], [0, 0, 0, 0], 'max')
            lv4 = bb.emit_te(topi.nn.flatten, lv3)
            lv5 = bb.emit_te(topi.nn.dense, lv4, linear0_weight, linear0_bias)
            lv6 = bb.emit_te(topi.nn.relu, lv5)
            lv7 = bb.emit_te(topi.nn.dense, lv6, linear1_weight, linear1_bias)
            #gv = bb.emit_te(topi.nn.dense, lv6, linear1_weight, linear1_bias)
            #gv = bb.emit_te(topi.nn.log_softmax, lv7, -1)
            gv = bb.emit_te(topi.nn.softmax, lv7, 1)
        bb.emit_func_output(gv)
    return bb.finalize()

def build_mod(mod):
    exec = relax.build(mod, "llvm")
    dev = tvm.cpu()
    vm = relax.VirtualMachine(exec, dev)
    return vm


def check_equivalence(mod, torch_model, test_loader):
    torch_model.eval()
    with torch.no_grad():
        rt_mod = build_mod(mod)
        for data, label in test_loader:
            data, label = data.cpu(), label.cpu()
            output_from_pytorch = torch_model(data).numpy()
            output_from_relax = rt_mod["main"](tvm.nd.array(data, tvm.cpu())).numpy()
            tvm.testing.assert_allclose(output_from_pytorch, output_from_relax, rtol=1e-4)


test_data = torchvision.datasets.FashionMNIST(
    "./data",
    download=True,
    train=False,
    transform=transforms.Compose([transforms.ToTensor()])
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

mod = create_model_via_emit_te()
print(mod.script())
torch_model = pytorch_model()

check_equivalence(mod, torch_model, test_loader)
print("Pass")

# Section 4. Transformation in End-to–End Models

'''
In Exercise 1, we learned how to transform a single TensorIR Function. It is similar to do that in an end-to-end model.

Compared with the batch matmul program, let's focus on a more challenging one: conv2d.

To begin with, let's introduce some new primitives:

compute_inline: It inlines a block into another to reduce memory usage and memory access.
fuse: The opposite for split. Fuse multiple axes. Here fuse is used together with parallel / vectorize / unroll to further increase parallelism.
'''

@T.prim_func
def before_inline(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


sch = tvm.tir.Schedule(before_inline)
sch.compute_inline(sch.get_block("B"))
print(sch.mod["main"].script())

@T.prim_func
def before_fuse(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0


sch = tvm.tir.Schedule(before_fuse)
i, j = sch.get_loops(sch.get_block("B"))
sch.fuse(i, j)
print(sch.mod["main"].script())

'''
Now we first create a schedule for the IRModule, and then transform the conv2d TensorIR function inside. Similar to Exercise 1, we provide you with a target function. But please note that, the target function does NOT serve as a "standard transformation answer" for several reasons:

it may not have the best performance on every hardware,
the original conv2d TensorIR implementation may vary, according to the Tensor Expression description you used in Section 2:
if you described the conv2d computation along with the bias computation in Tensor Expression, then there should be a block which calculates the bias at the end of target TensorIR function,
if you described conv2d and bias computation separately, or you used the conv2d provided by TOPI, then the target function should not have the bias block at the end. The original function of the target is generated by using TOPI conv2d.
'''

