using Tensornet.Math;
using Tensornet.Common;
using Tensornet.Native;

namespace Tensornet{
    public partial class Tensor<T>{
        public static Tensor<T> operator-(Tensor<T> self){
            Tensor<T> res = new Tensor<T>(new TensorLayout(self.TLayout, true));
            NegativeInternal(self, res);
            return res;
        }
        public static unsafe void NegativeInternal(Tensor<T> src, Tensor<T> dst){
            IntPtr status = NativeExecutor.Execute(NativeApi.Negative, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, IntPtr.Zero, Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
    }
}