{
  inputs.kernel-builder.url = "github:huggingface/kernels/f6bd904093ee680b77dbcee2b9c9292bb5f2a722";
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs { inherit self; path = ./.; };
}
