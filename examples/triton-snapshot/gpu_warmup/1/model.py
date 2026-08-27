import numpy as np
import cupy as cp

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.w1 = cp.random.standard_normal((1024, 2048), dtype=cp.float32) * 0.01
        self.w2 = cp.random.standard_normal((2048, 1024), dtype=cp.float32) * 0.01
        warm = cp.zeros((1, 1024), dtype=cp.float32)
        for _ in range(32):
            hidden = cp.maximum(warm @ self.w1, 0)
            warm = hidden @ self.w2
        cp.cuda.Stream.null.synchronize()

    def execute(self, requests):
        responses = []
        for request in requests:
            tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            values = cp.asarray(tensor.as_numpy(), dtype=cp.float32)
            hidden = cp.maximum(values @ self.w1, 0)
            result = cp.asnumpy(hidden @ self.w2).astype(np.float32)
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("OUTPUT", result)]
                )
            )
        return responses
