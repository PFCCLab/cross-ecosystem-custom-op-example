#include <ATen/Functions.h>
#include <torch/python.h>
#include <torch/library.h>
#include <pybind11/detail/common.h>

/**
 * Part1: Kernel implementation
 * 
 * This part is framework-agnostic and can be reused in other frameworks
 * without any modification.
**/
template<typename T>
void muladd_cpu_impl(const T* a_ptr, const T* b_ptr, T c, T* result_ptr, int64_t numel) {
  for (int64_t i = 0; i < numel; i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
  }
}

/**
 * Part2: Wrapper function for PyTorch
 * 
 * This part is PyTorch-specific and maybe needs to be modified
 * 
 * PaddlePaddle provides similar APIs to make this can be compatible with PaddlePaddle.
 * 
 * For some APIs that are not available in PaddlePaddle, you can get the original PaddlePaddle
 * structs and use PaddlePaddle specific APIs to implement similar functionalities.
 * 
 * For example:
 * 
 * ```cpp
 * at::IntArrayRef sizes = {2, 3, 4};
 * at::Tensor reshaped_tensor = x.reshape(sizes);
 * ```
 * 
 * If Tensor.reshape is not available in PaddlePaddle, you can use PaddlePaddle specific APIs to get the
 * sizes and reshape the tensor.
 * 
 * ```cpp
 * at::IntArrayRef sizes = {2, 3, 4};
 * auto paddle_tensor = x._PD_GetInner();  // Get the original PaddlePaddle tensor
 * auto paddle_sizes = shape._PD_ToPaddleIntArray();  // Convert to PaddlePaddle specific IntArray
 * auto paddle_reshaped_tensor = paddle::experimental::reshape(paddle_tensor, sizes);  // Reshape using PaddlePaddle API
 * at::Tensor reshaped_tensor(paddle_reshaped_tensor);  // Wrap back to at::Tensor
**/
at::Tensor muladd_cpu(at::Tensor a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  muladd_cpu_impl<float>(a_ptr, b_ptr, static_cast<float>(c), result_ptr, result.numel());
  return result;
}

/**
 * Part3: Binding code to register the operator with PyTorch
 * 
 * This part is PyTorch-specific and maybe needs to be modified
 * 
 * PaddlePaddle provides similar APIs to register custom operators.
 * Generally, you don't need to modify the schema definition.
**/
extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
  The import from Python will load the .so consisting of this file
  in this extension, so that the TORCH_LIBRARY static initializers
  below are run. */
PyObject* PyInit_extension_cpp(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "extension_cpp", /* name of module */
      NULL, /* module documentation, may be NULL */
      -1,   /* size of per-interpreter state of the module,
              or -1 if the module keeps state in global variables. */
      NULL, /* methods */
  };
  return PyModule_Create(&module_def);
}
}

TORCH_LIBRARY(extension_cpp, m) {
  m.def("muladd_cpp(Tensor a, Tensor b, float c) -> Tensor");
}

TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) { m.impl("muladd_cpp", &muladd_cpu); }
