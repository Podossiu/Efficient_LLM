import os
import warnings
import torch

__all__ = ["LayerNormMLP"]

# TODO: replace with my activation function kernel
def _act_func(activation: str):
    funcs = {
            'gelu': (tex.gelu, tex.dgelu),
            'relu', (tex.relu, tex.drelu),
            'geglu': (tex.geglu, tex.dgeglu),
            'reglu': (tex.reglu, tex.dreglu),
            'swiglu': (tex.swiglu, tex.dswiglu),
    }
    if activation not in funcs:
        raise NotImplementedError("Activation type " + activation + " is not supported!")
    return funcs[activation]

# Layer Norm -> FC1 -> Act -> FC2를 모두 퓨전?
# UB: User buffer communication backend
class _LayerNormMLP(torch.autograd.Function):
    """LayerNormMLP semi-top level module
    Calls custom cuda extensions.
    """
    
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        fc1_weight: torch.Tensor,
        # TODO: 확인 필요
        fc1_weight_fp8: Union[torch.Tensor, None],
        fc1_weight_t_fp8: Union[torch.Tensor, None],
        fc1_bias: torch.Tensor,
        use_fc1_bias: bool,
        fc2_weight: torch.Tensor,
        fc2_weight_fp8: Union[torch.Tensor, None],
        fc2_weight_t_fp8: Union[torch.Tensor, None],
        fc2_bias: torch.Tensor,
        use_fc2_bias: bool,
        eps: float,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        fuse_w_grad_accumulation: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        return_layernorm_output: bool,
        bias_gelu_nvfusion: bool,
        set_parallel_mode: bool,
        is_grad_enabled: bool,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
        activation: str,
        normalization: str,
        primary_weights_in_fp8: bool,
        ub_bulk_wgrad: bool,
        ub_bulk_dgrad: bool,
        ub_split_rs: bool,
        ub_atomic_gemm_rs: bool,
        ub_split_ag: bool,
        ub_atomic_gemm_ag: bool,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        # Make sure input dimensions are compatible
        in_features = ln_weight.numel()
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))
        if fp8:
            assert_dim_for_fp8_exec(inputmat)
            assert_dim_for_fp8_exec(fc1_weight)
            assert_dim_for_fp8_exec(fc2_weight)
        
        update_fp8_weights = is_first_microbatch is None or is_first_microbatch
        
        activation_func = _act_func(activation_func)[0]

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        if ln_bias is not None:
            ln_bias = cast_if_needed(ln_bias, activation_dtype)
        
        # TODO: 확인 필요
        # TP group: Tensor Parallel procesing group
        # ub_atomic? ub_split_ag?
        if ub_split_ag or ub_atomic_gemm_ag:
            tp_word_size = get_distributed_world_size(tp_group)
            if tp_world_size == 1 or (not is_grad_enabled) or return_layernorm_output:
                ub_split_ag = False
                ub_atomic_gemm_ag = False
        ub_overlap_ag = ub_split_ag or ub_atomic_gemm_ag
        
        if ub_overlap_ag:
            ub_obj_lnout = get_ub("fc1_fprop")
            ln_out = ub_obj_lnout.get_ubuf_output(0)
        else:
            ln_out_dtype = torch.uint8 if (fp8 and not return_layernorm_output) else inputmat.dtype
            ln_out = torch.empty_like(inputmat, dtype = ln_out_dtype)

        if ub_split_rs or ub_atomic_gemm_rs:
            tp_world_size = get_distributed_world_size(tp_group)
            if tp_world_size == 1:
                ub_split_rs = False
                ub_atomic_gemm_rs = False

        if ub_atomic_gemm_rs or ub_atomic_gemm_ag:
            assert fp8, "AtomicGemm overlap supported only for FP8 GEMM."


        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor = True)
        
        ln_out, mu, rsigma = _apply_normalization(inputmat,
                                                  ln_out,
                                                  ln_weight,
                                                  ln_bias,
                                                  eps,
                                                  fp8 and not return_layernorm_output,
                                                  fp8_meta,
                                                  normalization,
                                                  fwd_ln_sm_margin,
                                                  zero_centered_gamma,
                                                  is_grad_enabled)
        
        # If reisudal connection is after LN, we need 'ln_out'
        # tensor in higher precision, this comes at the cost 
        # of an extra fp8 cast
        if return_layernorm_output:
            ln_out_return = ln_out
            if fp8:
                ln_out = tex.cast_to_fp8(
                        ln_out,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_INPUT,
                        fp8_dtype_forward,
                )
                # 175 line



# TODO: TransformerEngineBaseModule 변경 
class LayerNormMLP(TransformerEngineBaseModule):
    r"""
    Applies layer normalization on the input followed by the MLP module, consisting of
    2 successive linear transformations, separated by the GeLU activation.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    ffn_hidden_size : int
                     intermediate size to which input samples are projected.
    eps : float, default = 1e-5
         a value added to the denominator of layer normalization for numerical stability.
    bias : bool, default = `True`
          if set to `False`, the FC1 and FC2 layers will not learn an additive bias.
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    activation : str, default = 'gelu'
          activation function used.
          Options: 'gelu', 'geglu', 'relu', 'reglu', 'squared_relu', 'swiglu'.
    init_method : Callable, default = `None`
                 used for initializing FC1 weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
                              used for initializing FC2 weights in the following way:
                              `output_layer_init_method(weight)`. When set to `None`, defaults to
                              `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module
                             is taken post layernorm.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.

    Parallelism parameters
    ----------------------
    set_parallel_mode : bool, default = `False`
                      if set to `True`, FC1 is used as Column Parallel and FC2 is used as Row
                      Parallel as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias for FC2, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    seq_length: int
               sequence length of input samples. Needed for JIT Warmup, a technique where jit fused
               functions are warmed up before training to ensure same kernels are used for forward
               propogation and activation recompute phase.
    micro_batch_size: int
                     batch size per training step. Needed for JIT Warmup, a technique where jit
                     fused functions are warmed up before training to ensure same kernels are
                     used for forward propogation and activation recompute phase.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        return_bias: bool = False,
        get_rng_state_tracker: Optional[Callable] = None,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        normalization: str = 'LayerNorm',
        activation : str = "gelu",
        output_layer_init_method: Optional[Callable] = None,
        fuse_wgrad_accumulation: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_layernorm_output: bool = False,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        set_parallel_mode: bool = False,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "cuda",
        ub_bulk_wgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_split_rs: bool = False,
        ub_atomic_gemm_rs: bool = False,
        ub_split_ag: bool = False,
        ub_atomic_gemm_ag: bool = False,
    ) -> None:

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.normalization = normalization
        assert normalization in ['LayerNorm', 'RMSNorm'], "Unsupported normalization type!"
        self.use_bias = bias
        self.activation = activation
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.return_layernorm_output = return_layernorm_output
        # bias gelu fusion ? 뭘까?
        self.bias_gelu_nvfusion = (bool(int(os.getenv("NVTE_BIAS_GELU_NVFUSION", "1"))) and
                                   self.activation == 'gelu')
        self.set_parallel_mode = set_parallel_mode
        self.zero_centered_gamma = zero_centered_gamma
        # TODO: weights in fp8 global state manager알아야해 
        self.primary_weights_in_fp8 = FP8GlobalStateManager.with_fp8_parameters()
        self.ub_bulk_wgrad = ub_bulk_wgrad
        self.ub_bulk_dgrad = ub_bulk_dgrad
        self.ub_split_rs = ub_split_rs
        self.ub_split_ag = ub_split_ag
        self.ub_atomic_gemm_rs = ub_atomic_gemm_rs
        self.ub_atomic_gemm_ag = ub_atomic_gemm_ag
        
        # TODO: User buffer communication 공부
        if (ub_bulk_wgrad # pylint: disable=too-many-boolean-expressions
            or ub_bulk_dgrad
            or ub_split_rs
            or ub_split_ag
            or ub_atomic_gemm_rs
            or ub_atomic_gemm_ag):
            assert (
                tex.userbuf_comm_available()
            ), "Userbuffer communication backend not available."

        if ub_atomic_gemm_rs or ub_atomic_gemm_ag:
            warnings.warn(
                "Atomic gemm uses a beta API from cublas and is not tested for all use cases."
            )

        # Parallel Communication
        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()
        
        if init_method is None:
            init_method = get_default_init_method() # FC1 weight initializer
        if output_layer_init_method is None:
            output_layer_init_method = get_default_init_method() # FC2 weight initializer
        

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel
        self.size_per_partition = divide(ffn_hidden_size, self.tp_size)

        # LN init
        self.eps = eps
        self.layer_norm_weight = Parameter(
            torch.empty(hidden_size, device=device, dtype=params_dtype)
        )
        setattr(self.layer_norm_weight, "sequence_parallel", self.sequence_parallel)
        if self.normalization != "RMSNorm":
            self.layer_norm_bias = Parameter(
                torch.empty(hidden_size, device=device, dtype=params_dtype)
            )
            setattr(self.layer_norm_bias, "sequence_parallel", self.sequence_parallel)
        else:
            self.layer_norm_bias = None
        self.reset_layer_norm_parameters()

        if self.activation in ['rgelu', 'geglu', 'swiglu']:
            # for gating features
            fc1_output_features = 2 * sef.size_per_partition
        else:
            fc1_output_features = self.size_per_partition

        # FC1 init
        # 1. weight를 param_dtype에 맞춰서 빈 텐서 생성 
        fc1_temp_weight = torch.empty(
            fc1_output_features, hidden_size, device=device, dtype=params_dtype)
        
        # initialize fc1 temp weight
        initialize_affine_weight_gpu(
            fc1_temp_weight,
            init_method,
            get_rng_state_tracker,
            set_tp_attributes=False,
        )
        
        # primary weight가 fp8? 이거는 학습할 때 쓰는 weight가 fp8이라는 거 같은데 
        if self.primary_weights_in_fp8:
            self.init_fp8_metadata(num_gemms=2)
            self.fp8_meta["update_amax_and_scale_fwd"] = True

            fc1_temp_weight = Float8Tensor.to_float8(
                fc1_temp_weight,
                fp8_meta=self.fp8_meta,
                fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
            )

        self.fc1_weight = Parameter(fc1_temp_weight)
        set_tensor_model_parallel_attributes(self.fc1_weight, True, 0, 1)
        self.fp8_weight_shapes.append(self.fc1_weight.shape)

        if self.use_bias:
            self.fc1_bias = Parameter(
                torch.empty(fc1_output_features, device=device, dtype=params_dtype)
            )
            set_tensor_model_parallel_attributes(self.fc1_bias, True, 0, 1)
        else:
            self.fc1_bias = torch.Tensor().to(dtype=params_dtype, device=device)

        with torch.no_grad():
            self.fc1_bias.zero_()
        # FC2 init
        fc2_temp_weight = torch.empty(
            hidden_size, self.size_per_partition, device=device, dtype=params_dtype)

        initialize_affine_weight_gpu(
            fc2_temp_weight,
            output_layer_init_method,
            get_rng_state_tracker,
            set_tp_attributes=False,
        )

        if self.primary_weights_in_fp8:
            fc2_temp_weight = Float8Tensor.to_float8(
                fc2_temp_weight,
                fp8_meta=self.fp8_meta,
                fp8_meta_index=tex.FP8FwdTensors.GEMM2_WEIGHT,
            )

        self.fc2_weight = Parameter(fc2_temp_weight)
        set_tensor_model_parallel_attributes(self.fc2_weight, True, 1, 1)
        self.fp8_weight_shapes.append(self.fc2_weight.shape)

        if self.use_bias:
            self.fc2_bias = Parameter(
                torch.empty(hidden_size, device=device, dtype=params_dtype)
            )
            # RPL
            if self.set_parallel_mode:
                setattr(self.fc2_bias, "sequence_parallel", sequence_parallel)
        else:
            self.fc2_bias = torch.Tensor().to(dtype=params_dtype, device=device)
	
        # TODO: What is RPL?
        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.set_parallel_mode and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

        with torch.no_grad():
            self.fc2_bias.zero_()

        if self.bias_gelu_nvfusion:
            set_jit_fusion_options()
            if seq_length and micro_batch_size:
                warmup_jit_bias_gelu_all_dtypes(
                    self.size_per_partition, seq_length, micro_batch_size
                )
        # These many SMs are subtracted from the total SM count when calling forward
        # and backward LayerNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with LN.
        self.fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_ln_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        if not self.zero_centered_gamma:
            init.ones_(self.layer_norm_weight)
        else:
            init.zeros_(self.layer_norm_weight)
        if self.layer_norm_bias is not None:
            init.zeros_(self.layer_norm_bias)


    def get_fp8_weights_scratchpad(
            self,
            is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        """
        Fetch the fp8 weight tensor placeholders if they exist (when
        'is_first_microbatch' is not 'None') or return empty fp8 weight
        tensors (if 'is_first_microbatch is None)
        so, if is_first_microbatch is None -> return empty fp8 weight
            if is_first_microbatch is not None -> return fp8 weight placeholders
        """
        if not self.fp8 or self.primary_weights_in_fp8:
            return [None, None, None, None]
        
        if is_first_microbatch is None:
            fp8_weight_tensors = self.get_fp8_weights_empty_tensors(
                    is_first_microbatch
            )
        else:
            # These persistent weight placeholders should've been created in
            # 'set_fp8_weights' method
            fp8_weight_tensors = [self.weight1_fp8, self.weight1_t_fp8,
                                  self.weight2_fp8, self.weight2_t_fp8]
        return fp8_weight_tensors

    
    @no_torch_dynamo()
    def forward(
        self, inp: torch.Tensor, is_first_microbatch: Optional[bool] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply layer normalization to the input followed by a feedforward network (MLP Block).

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. ***Between the microbatches of the same minibatch
                             the model weights are not updated.*** Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                               #첫 번째 micro batch는그 자체로 gradient기 때문에 누적은 안해도됨
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        """
        with self.prepare_forward(inp, is_first_microbatch, num_gemms=2) as inp:
            assert self.fp8 or not self.primary_weights_in_fp8, \
                   "Need to run inside fp8_autocast region when weights are stored in FP8."
            # Fetch the fp8 weights placeholders (for linear/gemm)
            weight1_fp8, weight1_t_fp8, weight2_fp8, weight2_t_fp8 = \
                self.get_fp8_weights_scratchpad(
                        is_first_microbatch
                )

            if torch.is_grad_enabled():
                fwd_fn = _LayerNormMLP.apply
                args = []
            else:
                fwd_fn = _LayerNormMLP.forward
                args = [None]
            args += (
                inp,
                self.layer_norm_weight,
                self.layer_norm_bias,
                self.fc1_weight,
                weight1_fp8,
                weight1_t_fp8,
                self.fc1_bias,
                self.use_bias,
                self.fc2_weight,
                weight2_fp8,
                weight2_t_fp8,
                self.fc2_bias,
                self.apply_bias and not self.gemm_bias_unfused_add,
                self.eps,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.fp8_meta,
                self.fuse_wgrad_accumulation,
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.return_layernorm_output,
                self.bias_gelu_nvfusion,
                self.set_parallel_mode,
                torch.is_grad_enabled(),
                self.fwd_ln_sm_margin,
                self.bwd_ln_sm_margin,
                self.zero_centered_gamma,
                self.activation,
                self.normalization,
                self.primary_weights_in_fp8,
                self.ub_bulk_wgrad,
                self.ub_bulk_dgrad,
                self.ub_split_rs,
                self.ub_atomic_gemm_rs,
                self.ub_split_ag,
                self.ub_atomic_gemm_ag,
            )

            out = fwd_fn(*args)

        if self.return_layernorm_output:
            out, ln_out = out

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(self.fc2_bias, self.activation_dtype)

        if self.return_bias:
            if self.return_layernorm_output:
                return out, cast_if_needed(self.fc2_bias, self.activation_dtype), ln_out
            return out, cast_if_needed(self.fc2_bias, self.activation_dtype)
        if self.return_layernorm_output:
            return out, ln_out
        return out




