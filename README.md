# Cross Ecosystem Custom Operator Example

This repository contains an example of a PyTorch custom operator that can be migrated to PaddlePaddle. The custom operator performs a simple element-wise multiplication and addition operation.

## Usage in PyTorch

### Prerequisites

- PyTorch installed, refer to [PyTorch Installation](https://pytorch.org/get-started/locally/) for instructions.

### Building and Installing the Custom Operator

```bash
pip install . --no-build-isolation
```

### Running the PyTorch Example

```bash
python test.py
```

## Migrating to PaddlePaddle

### Prerequisites

- PaddlePaddle 3.3+ or nightly version installed, refer to [PaddlePaddle Installation](https://www.paddlepaddle.org.cn/install/quick) for instructions.

### Building and Installing the PaddlePaddle Custom Operator

- Step 1: Modify the `setup.py` file to use PaddlePaddle's C++ extension utilities.

  ```diff
  +import paddle
  +paddle.compat.enable_torch_proxy()  # Enable torch proxy globally

  from setuptools import setup, find_packages
  # This torch extension will be replaced by PaddlePaddle's equivalent
  from torch.utils import cpp_extension

  setup(
      name="extension",
      packages=find_packages(include=['extension']),
      ext_modules=[
          cpp_extension.CUDAExtension(
              name="extension_cpp",
              sources=["csrc/muladd.cc"],
          )
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
  )
  ```

  In this case, you can simply add the line `paddle.compat.enable_torch_proxy()` to enable the torch proxy globally to replace the PyTorch C++ extension building utilities with PaddlePaddle's. For the other custom operators, you may need to manually replace the PyTorch C++ extension building utilities with PaddlePaddle's equivalent.

- Step 2: Try to build the custom operator with PaddlePaddle.

  ```bash
  pip install . --no-build-isolation
  ```

- Step 3: Follow the building error messages to modify the source code accordingly. You may need to replace PyTorch-specific APIs with PaddlePaddle equivalents.

  In this example, all PyTorch-specific APIs are covered by [PaddlePaddle C++ compatibility layer](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/api/include/compat), so no further code modification is needed.

  In other cases, you may need to refer to the [PaddlePaddle C++ Extension documentation](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html) to find the equivalent APIs. For example, if `Tensor.reshape` is not covered by the compatibility layer, you can manually implement it as follows:

  ```cpp
  // PyTorch version
  at::IntArrayRef sizes = {2, 3, 4};
  at::Tensor reshaped_tensor = x.reshape(sizes);
  ```

  We can replace it with PaddlePaddle equivalent:

  ```cpp
  // PaddlePaddle version
  at::IntArrayRef sizes = {2, 3, 4};
  auto paddle_tensor = x._PD_GetInner();  // Get the original PaddlePaddle tensor
  auto paddle_sizes = sizes._PD_ToPaddleIntArray();  // Convert to PaddlePaddle specific IntArray
  auto paddle_reshaped_tensor = paddle::experimental::reshape(paddle_tensor, paddle_sizes);  // Reshape using PaddlePaddle API
  at::Tensor reshaped_tensor(paddle_reshaped_tensor);  // Wrap back to at::Tensor
  ```

- Step 4: Run the PaddlePaddle example.

  Because the Python API layer is usually wrapped with PyTorch-specific code, we need to enable the torch proxy before importing the custom operator module.

  ```python
  import paddle
  paddle.compat.enable_torch_proxy(scope={"extension"})  # Enable torch proxy for the 'extension' module
  import extension

  x = paddle.tensor([1.0, 2.0, 3.0])
  y = paddle.tensor([4.0, 5.0, 6.0])
  z = 2.0
  result = extension.muladd(x, y, z)
  print(result)  # Expected output: tensor([ 6., 12., 20.])
  ```

- Step 5: Modify the python API layer if necessary.

  In this example, the existing Python API layer is compatible with PaddlePaddle after enabling the torch proxy, so no further modification is needed. However, in other cases, you may need to replace PyTorch-specific APIs with PaddlePaddle equivalents in the Python API layer as well.
