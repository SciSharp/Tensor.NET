using Numnet.Native;
using Numnet.Exceptions;

namespace Numnet{
    public partial class Tensor<T> where T : struct, IEquatable<T>, IConvertible{
        public Tensor<T> this[Tensor<bool> index]{
            get{
                Tensor<T> res = new Tensor<T>(new TensorLayout(TLayout, true));
                BoolIndexInternal(this, res, index);
                return res;
            }
        }
        private static unsafe void BoolIndexInternal(Tensor<T> src, Tensor<T> dst, Tensor<bool> index){
            if(index.TLayout.CanBroadCastTo(dst.TLayout)){
                var indexLayout = index.TLayout.Broadcast(dst.TLayout);
                IntPtr status = NativeExecutor.Execute(NativeApi.BoolIndex, src.TMemory, index.TMemory, dst.TMemory, src.TLayout, indexLayout, dst.TLayout, IntPtr.Zero, Tensor<T>.Provider);
                NativeStatus.AssertOK(status);
            }
            else{
                throw new MismatchedShapeException($"Cannot broadcast from {(index.TLayout as TensorShape)} to {(dst.TLayout as TensorShape)}.");
            }
        }
    }
}