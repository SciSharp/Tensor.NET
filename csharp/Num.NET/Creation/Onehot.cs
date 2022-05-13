using Numnet.Native.Param;
using Numnet.Native;

namespace Numnet{
    public static partial class Tensor{
        public static Tensor<T> Onehot<T>(Tensor<T> src, int maxValue) where T : struct, IConvertible, IEquatable<T>{
            Tensor<T> res = new Tensor<T>(DeduceOnehotLayout(src.TLayout, maxValue));
            res.TLayout.InitContiguousLayout();
            OnehotInternal(src, res, maxValue);
            return res;
        }
        private unsafe static void OnehotInternal<T>(Tensor<T> src, Tensor<T> dst, int maxValue) where T : struct, IEquatable<T>, IConvertible{
            OnehotParam p = new OnehotParam() { maxValue = maxValue };
            IntPtr status = NativeExecutor.Execute(NativeApi.Onehot, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, new IntPtr(&p), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
        private static TensorLayout DeduceOnehotLayout(TensorLayout layout, int maxValue){
            TensorLayout res = new TensorLayout();
            res.DType = layout.DType;
            res.NDim = layout.NDim + 1;
            for (int i = 0; i < layout.NDim; i++) {
                res.Shape[i] = layout.Shape[i];
            }
            res.Shape[res.NDim - 1] = maxValue + 1;
            return res;
        }
    }
}