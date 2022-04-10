using Numnet.Common;

namespace Numnet{
    public partial class Tensor{
        // public Tensor<T> Reshape(int[] shape){
        //     return new Tensor<T>(this.TMemory, this.TLayout.Reshape(new TensorShape(shape), false));
        // }
        // public Tensor<T> Reshape(TensorShape shape){
        //     return new Tensor<T>(this.TMemory, this.TLayout.Reshape(shape, false));
        // }
        public Tensor Reshape(int[] shape){
            return new Tensor(this.TMemory, this.TLayout.Reshape(new TensorShape(shape), false));
        }
        public Tensor Reshape(TensorShape shape){
            return new Tensor(this.TMemory, this.TLayout.Reshape(shape, false));
        }

        public void ReshapeTo(int[] shape){
            TLayout = TLayout.Reshape(new TensorShape(shape), false);
        }
        public void ReshapeTo(TensorShape shape){
            TLayout = TLayout.Reshape(shape, false);
        }
    }
}