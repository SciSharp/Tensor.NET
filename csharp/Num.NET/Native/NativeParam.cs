using Numnet.Tensor.Base;
using Numnet.Exceptions;

namespace Numnet.Native.Param{
    struct MatmulParam{
        
    }

    struct Permute{
        int[] dims;
        public Permute(params int[] dims){
            if(dims.Length > TensorShape.MAX_NDIM){
                throw new DimExceedException(dims.Length);
            }
            this.dims = new int[dims.Length];
            dims.CopyTo(this.dims.AsSpan());
        }
    }
}