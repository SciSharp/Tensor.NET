using Numnet.Common;

namespace Numnet{
    public partial class Tensor{
        // public Tensor<T> BroadCast(TensorShape targetShape){
        //     return new Tensor<T>(this.TMemory, this.TLayout.BroadcastTo(targetShape));
        // }

        // public Tensor<T> BroadCast(int[] targetShape){
        //     return new Tensor<T>(this.TMemory, this.TLayout.BroadcastTo(new TensorShape(targetShape)));
        // }
        public Tensor Broadcast(Span<int> targetShape){
            return new Tensor(this.TMemory, this.TLayout.Broadcast(new TensorShape(targetShape)));
        }

        public Tensor Broadcast(TensorShape targetShape){
            return new Tensor(this.TMemory, this.TLayout.Broadcast(targetShape));
        }

        public Tensor Broadcast(int[] targetShape){
            return new Tensor(this.TMemory, this.TLayout.Broadcast(new TensorShape(targetShape)));
        }
        public virtual void BroadcastTo(TensorShape targetShape){
            TLayout.BroadcastInplace(targetShape);
        }
        public virtual void BroadcastTo(int[] targetShape){
            TLayout.BroadcastInplace(new TensorShape(targetShape));
        }
    }
}