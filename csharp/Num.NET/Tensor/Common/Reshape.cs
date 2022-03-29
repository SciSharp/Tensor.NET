using Numnet.Tensor;

namespace Numnet.Tensor{
    public sealed partial class Tensor<T>{
        public Tensor<T> Reshape(int[] shape){
            return new Tensor<T>(this.TMemory, this.TLayout.Reshape(shape, false));
        }
    }
}