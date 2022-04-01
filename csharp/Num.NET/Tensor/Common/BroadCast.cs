namespace Numnet.Tensor{
    public sealed partial class Tensor<T>{
        public Tensor<T> BroadCast(int[] targetShape){
            return new Tensor<T>(this.TMemory, this.TLayout.BroadcastTo(targetShape));
        }

        public void BroadCastTo(int[] targetShape){
            this.TLayout.BroadcastInplace(targetShape);
        }
    }
}