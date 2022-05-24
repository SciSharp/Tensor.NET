
namespace Tensornet.Common{
    internal static class InplaceOperation{
        public static void Execute<T>(Tensor<T> inp, Func<T, T> operation) 
            where T : struct, IEquatable<T>, IConvertible{
            int idxOffset = inp.TLayout.NDim - TensorLayout.MAX_NDIM;
            Span<T> spanInp = inp.AsSpan();
            for (int n = 0; n < (idxOffset >= 0 ? inp.TLayout.Shape[idxOffset] : 1); n++){
                int nOffsetInp = idxOffset >= 0 ? n * inp.TLayout.Stride[idxOffset] : 0;
                int nOffsetinp = idxOffset >= 0 ? n * inp.TLayout.Stride[idxOffset] : 0;
                for (int c = 0; c < (idxOffset >= -1 ? inp.TLayout.Shape[idxOffset + 1] : 1); c++)
                {
                    int ncOffsetInp = (idxOffset >= -1 ? c * inp.TLayout.Stride[idxOffset + 1] : 0) + nOffsetInp;
                    int ncOffsetinp = (idxOffset >= -1 ? c * inp.TLayout.Stride[idxOffset + 1] : 0) + nOffsetinp;
                    for (int h = 0; h < (idxOffset >= -2 ? inp.TLayout.Shape[idxOffset + 2] : 1); h++){
                        int nchOffsetInp = (idxOffset >= -2 ? h * inp.TLayout.Stride[idxOffset + 2] : 0) + ncOffsetInp;
                        int nchOffsetinp = (idxOffset >= -2 ? h * inp.TLayout.Stride[idxOffset + 2] : 0) + ncOffsetinp;
                        for (int w = 0; w < (idxOffset >= -3 ? inp.TLayout.Shape[idxOffset + 3] : 1); w++){
                            spanInp[nchOffsetinp + w * inp.TLayout.Stride[idxOffset + 3]] = 
                                operation(spanInp[nchOffsetInp + w * inp.TLayout.Stride[idxOffset + 3]]);
                        }
                    }
                }
            }
        }

        public static void Execute<TInput, TRefer>(Tensor<TInput> inp, Tensor<TRefer> refer, Func<TInput, TRefer, TInput> operation) 
            where TInput : struct, IEquatable<TInput>, IConvertible 
            where TRefer : struct, IEquatable<TRefer>, IConvertible{
            int idxOffset = inp.TLayout.NDim - TensorLayout.MAX_NDIM;
            Span<TRefer> spanRefer = refer.AsSpan();
            Span<TInput> spanInp = inp.AsSpan();
            for (int n = 0; n < (idxOffset >= 0 ? refer.TLayout.Shape[idxOffset] : 1); n++){
                int nOffsetInp = idxOffset >= 0 ? n * inp.TLayout.Stride[idxOffset] : 0;
                int nOffsetRefer = idxOffset >= 0 ? n * refer.TLayout.Stride[idxOffset] : 0;
                for (int c = 0; c < (idxOffset >= -1 ? refer.TLayout.Shape[idxOffset + 1] : 1); c++)
                {
                    int ncOffsetInp = (idxOffset >= -1 ? c * inp.TLayout.Stride[idxOffset + 1] : 0) + nOffsetInp;
                    int ncOffsetRefer = (idxOffset >= -1 ? c * refer.TLayout.Stride[idxOffset + 1] : 0) + nOffsetRefer;
                    for (int h = 0; h < (idxOffset >= -2 ? refer.TLayout.Shape[idxOffset + 2] : 1); h++){
                        int nchOffsetInp = (idxOffset >= -2 ? h * inp.TLayout.Stride[idxOffset + 2] : 0) + ncOffsetInp;
                        int nchOffsetrefer = (idxOffset >= -2 ? h * refer.TLayout.Stride[idxOffset + 2] : 0) + ncOffsetRefer;
                        for (int w = 0; w < (idxOffset >= -3 ? refer.TLayout.Shape[idxOffset + 3] : 1); w++){
                            spanInp[nchOffsetInp + w * inp.TLayout.Stride[idxOffset + 3]] = 
                                operation(spanInp[nchOffsetInp + w * inp.TLayout.Stride[idxOffset + 3]], spanRefer[nchOffsetrefer + w * refer.TLayout.Stride[idxOffset + 3]]);
                        }
                    }
                }
            }
        }

        public static void Execute<T>(Tensor<T> inp, Tensor<bool> condition, Tensor<T> refer) 
            where T : struct, IEquatable<T>, IConvertible {
            int idxOffset = inp.TLayout.NDim - TensorLayout.MAX_NDIM;
            Span<T> spanRefer = refer.AsSpan();
            Span<T> spanInp = inp.AsSpan();
            Span<bool> spanCondition = condition.AsSpan();
            for (int n = 0; n < (idxOffset >= 0 ? refer.TLayout.Shape[idxOffset] : 1); n++){
                int nOffsetInp = idxOffset >= 0 ? n * inp.TLayout.Stride[idxOffset] : 0;
                int nOffsetRefer = idxOffset >= 0 ? n * refer.TLayout.Stride[idxOffset] : 0;
                int nOffsetCondition = idxOffset >= 0 ? n * condition.TLayout.Stride[idxOffset] : 0;
                for (int c = 0; c < (idxOffset >= -1 ? refer.TLayout.Shape[idxOffset + 1] : 1); c++)
                {
                    int ncOffsetInp = (idxOffset >= -1 ? c * inp.TLayout.Stride[idxOffset + 1] : 0) + nOffsetInp;
                    int ncOffsetRefer = (idxOffset >= -1 ? c * refer.TLayout.Stride[idxOffset + 1] : 0) + nOffsetRefer;
                    int ncOffsetCondition = (idxOffset >= -1 ? c * condition.TLayout.Stride[idxOffset + 1] : 0) + nOffsetCondition;
                    for (int h = 0; h < (idxOffset >= -2 ? refer.TLayout.Shape[idxOffset + 2] : 1); h++){
                        int nchOffsetInp = (idxOffset >= -2 ? h * inp.TLayout.Stride[idxOffset + 2] : 0) + ncOffsetInp;
                        int nchOffsetrefer = (idxOffset >= -2 ? h * refer.TLayout.Stride[idxOffset + 2] : 0) + ncOffsetRefer;
                        int nchOffsetCondition = (idxOffset >= -2 ? h * condition.TLayout.Stride[idxOffset + 2] : 0) + ncOffsetCondition;
                        for (int w = 0; w < (idxOffset >= -3 ? refer.TLayout.Shape[idxOffset + 3] : 1); w++){
                            if(spanCondition[nchOffsetCondition + w * condition.TLayout.Stride[idxOffset + 3]]){
                                spanInp[nchOffsetInp + w * inp.TLayout.Stride[idxOffset + 3]] = spanRefer[nchOffsetrefer + w * refer.TLayout.Stride[idxOffset + 3]];
                            }
                        }
                    }
                }
            }
        }
    }
}

namespace Tensornet{
    public partial class Tensor<T>{
        /// <summary>
        /// Apply an function on each element of the current tensor inplace.
        /// Note that the modification will be made on the current tensor directly.
        /// </summary>
        /// <param name="operation"></param>
        public void ForEachInplace(Func<T, T> operation){
            Common.InplaceOperation.Execute(this, operation);
        }
    }
}