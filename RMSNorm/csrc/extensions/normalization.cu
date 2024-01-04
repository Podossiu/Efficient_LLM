/*************************************************************************
* Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
*
* See LICENSE for license information.
************************************************************************/

/***
https://github.com/NVIDIA/TransformerEngine/blob/fad3044bde1547eae9543a6a3f80401e59bb629e/transformer_engine/pytorch/csrc/extensions/normalization.cu
***/

#include "extensions.h"
#include <ATen/Aten.h>


std::vector<at::Tensor> rmsnorm_fwd_fp8(const at::Tensor &input,
					const at::Tensor &weight,
					float eps,
					at::Tensor scale,
					at::Tensor amax,
					at::Tensor scale_inv,
					transformer_engine::DType otype,
					const int sm_margin,
					const bool zero_centered_gamma
) {

}
std::vector<at::Tensor> rmsnorm_fwd(const at::Tensor &input,
		 		    const at::Tensor &weight,
				    float eps,
				    const int sm_margin,
				    const bool zero_centered_gamma
) {

    aten::ScalarType itype = input.scalar_type();
    auto ln_out = at::empty_like(input, at::CUDA(itype));

    return rmsnorm_fwd_noalloc(input, weight, ln_out, eps,
		    	       sm_margin, zero_centered_gamma);
}

std::vector<at::Tensor> rmsnorm_fwd_noalloc(const at::Tensor &input,
					    const at::Tensor &weight,
					    at::Tensor ln_out,
					    float eps,
					    const int sm_margin,
					    const bool zero_centered_gamma
) {

    aten::ScalarType itype = input.scalar_type();

    return rmsnorm_fwd_fp8_noalloc(input, weight, eps, at::Tensor(),
		                   ln_out, at::Tensor(), at::Tensor(),
				   itype, sm_margin, zero_centered_gamma);
}

at::Tensor rmsnorm_fwd_inf(const at::Tensor &input,
			   const at::Tensor &weight,
			   float eps,
			   const bool zero_centered_gamma
) {
    // This is a specialized version of rmsnorm_fwd, optimized for inference,
    // which only return the normalized output.
    std::vector<at::Tensor> out = rmsnorm_fwd(input, weight, eps, 0, zero_centered_gamma);
    return out[0];
}

