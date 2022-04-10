using Numnet.Common;
using Numnet.Native;
using Numnet.Exceptions;
using Numnet.Native.Param;

namespace Numnet.Manipulation{
    public static class PermuteExtension{

        unsafe public static Tensor Permute(this Tensor src, int[] dims)
        {
            Tensor res = new Tensor(DeduceLayout(src.TLayout, dims));
            res.TLayout.InitContiguousLayout();
            ExecutePermute(src, res, dims);
            return res;
        }
        private unsafe static void ExecutePermute(Tensor src, Tensor dst, int[] dims){
            fixed(int* pdims = dims){
                PermuteParam p = new PermuteParam() { dims = new IntPtr(pdims) };
                IntPtr status = Tensor.Execute(NativeApi.Permute, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, new IntPtr(&p), Tensor.Provider);
                NativeStatus.AssertOK(status);
            }
        }
        private static TensorLayout DeduceLayout(TensorLayout src, int[] dims){
            TensorLayout res = new TensorLayout();
            int duplicated = 0;
            res.DType = src.DType;
            res.NDim = src.NDim;
            for (int i = 0; i < src.NDim; i++) {
                if (dims[i] >= src.NDim) {
                    throw new InvalidParamException("Invalid param for permute.");
                }
                res.Shape[dims[i]] = src.Shape[i];
                if ((duplicated & (1 << dims[i])) != 0) {
                    throw new InvalidParamException("Duplicated index in the param of permute.");
                }
                duplicated += 1 << dims[i];
            }
            return res;
        }
    }
}