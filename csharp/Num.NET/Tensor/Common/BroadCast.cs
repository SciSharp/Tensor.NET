using Numnet.Tensor.Base;

namespace Numnet.Tensor{
    public sealed partial class Tensor<T>{
        public Tensor<T> BroadCast(TensorShape targetShape){
            return new Tensor<T>(this.TMemory, this.TLayout.BroadcastTo(targetShape));
        }

        public Tensor<T> BroadCast(int[] targetShape){
            return new Tensor<T>(this.TMemory, this.TLayout.BroadcastTo(new TensorShape(targetShape)));
        }
    }
}