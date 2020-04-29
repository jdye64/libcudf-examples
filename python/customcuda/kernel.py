import cudf
import customcuda._libxx.kernel as libkernel


# Base CustomKernel class to invoke the actual kernel
class CustomKernel(object):

    def __init__(self, kernel_name):
        self.name = kernel_name

    def run_kernel():
        print("Running custom kernel .... not yet implemented")


# Convert the weather rainfall totals from 1/10ths of mm to inches
class MMToInchesKernel(CustomKernel):

    def __init__(self):
        self.something = ""
