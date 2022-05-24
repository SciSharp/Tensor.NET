using Tensornet.Common;

namespace Tensornet{
    public partial class Tensor<T>{
        /// <summary>
        /// Broadcast the tensor to the given shape.
        /// Please note that the returned tensor shares the memory with the source tensor.
        /// </summary>
        /// <param name="targetShape"> The shape to broadcast to.</param>
        /// <returns></returns>
        public Tensor<T> Broadcast(Span<int> targetShape){
            return new Tensor<T>(this.TMemory, this.TLayout.Broadcast(new TensorShape(targetShape)));
        }
        /// <summary>
        /// Broadcast the tensor to the given shape.
        /// Please note that the returned tensor shares the memory with the source tensor.
        /// </summary>
        /// <param name="targetShape"> The shape to broadcast to.</param>
        /// <returns></returns>
        public Tensor<T> Broadcast(TensorShape targetShape){
            return new Tensor<T>(this.TMemory, this.TLayout.Broadcast(targetShape));
        }
        /// <summary>
        /// Broadcast the tensor to the given shape.
        /// Please note that the returned tensor shares the memory with the source tensor.
        /// </summary>
        /// <param name="targetShape"> The shape to broadcast to.</param>
        /// <returns></returns>
        public Tensor<T> Broadcast(int[] targetShape){
            return new Tensor<T>(this.TMemory, this.TLayout.Broadcast(new TensorShape(targetShape)));
        }
        /// <summary>
        /// Broadcast the tensor to the given shape in place.
        /// </summary>
        /// <param name="targetShape"> The shape to broadcast to.</param>
        public void BroadcastTo(TensorShape targetShape){
            TLayout.BroadcastInplace(targetShape);
        }
        /// <summary>
        /// Broadcast the tensor to the given shape in place.
        /// </summary>
        /// <param name="targetShape"> The shape to broadcast to.</param>
        public void BroadcastTo(int[] targetShape){
            TLayout.BroadcastInplace(new TensorShape(targetShape));
        }
    }
}