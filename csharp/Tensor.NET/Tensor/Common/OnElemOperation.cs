
namespace Tensornet.Common{
    internal static class OnElemOperation{
        public static Tensor<TResult> Execute<TInput, TResult>(Tensor<TInput> inp, Func<TInput, TResult> operation) 
            where TInput : struct, IEquatable<TInput>, IConvertible 
            where TResult : struct, IEquatable<TResult>, IConvertible{
            Tensor<TResult> res = new Tensor<TResult>(new TensorLayout(inp.TLayout, TensorTypeInfo.GetTypeInfo(typeof(TResult))._dtype));
            int idxOffset = res.TLayout.NDim - TensorLayout.MAX_NDIM;
            Span<TResult> spanRes = res.AsSpan();
            Span<TInput> spanInp = inp.AsSpan();
            for (int n = 0; n < (idxOffset >= 0 ? res.TLayout.Shape[idxOffset] : 1); n++){
                int nOffsetInp = idxOffset >= 0 ? n * inp.TLayout.Stride[idxOffset] : 0;
                int nOffsetRes = idxOffset >= 0 ? n * res.TLayout.Stride[idxOffset] : 0;
                for (int c = 0; c < (idxOffset >= -1 ? res.TLayout.Shape[idxOffset + 1] : 1); c++)
                {
                    int ncOffsetInp = (idxOffset >= -1 ? c * inp.TLayout.Stride[idxOffset + 1] : 0) + nOffsetInp;
                    int ncOffsetRes = (idxOffset >= -1 ? c * res.TLayout.Stride[idxOffset + 1] : 0) + nOffsetRes;
                    for (int h = 0; h < (idxOffset >= -2 ? res.TLayout.Shape[idxOffset + 2] : 1); h++){
                        int nchOffsetInp = (idxOffset >= -2 ? h * inp.TLayout.Stride[idxOffset + 2] : 0) + ncOffsetInp;
                        int nchOffsetRes = (idxOffset >= -2 ? h * res.TLayout.Stride[idxOffset + 2] : 0) + ncOffsetRes;
                        for (int w = 0; w < (idxOffset >= -3 ? res.TLayout.Shape[idxOffset + 3] : 1); w++){
                            spanRes[nchOffsetRes + w * res.TLayout.Stride[idxOffset + 3]] = 
                                operation(spanInp[nchOffsetInp + w * inp.TLayout.Stride[idxOffset + 3]]);
                        }
                    }
                }
            }
            return res;
        }
    }
}