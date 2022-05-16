using Tensornet.Common;

namespace Tensornet{
    public partial class Tensor<T>{
        public Tensor<T> Reshape(Span<int> shape){
            return new Tensor<T>(this.TMemory, this.TLayout.Reshape(new TensorShape(shape), false));
        }
        public Tensor<T> Reshape(int[] shape){
            return new Tensor<T>(this.TMemory, this.TLayout.Reshape(new TensorShape(shape), false));
        }
        public Tensor<T> Reshape(TensorShape shape){
            return new Tensor<T>(this.TMemory, this.TLayout.Reshape(shape, false));
        }

        public void ReshapeTo(int[] shape){
            TLayout = TLayout.Reshape(new TensorShape(shape), false);
        }
        public void ReshapeTo(TensorShape shape){
            TLayout = TLayout.Reshape(shape, false);
        }
    }
}