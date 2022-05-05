
namespace Numnet{
    public partial class Tensor<T>{
        public Tensor<T> Unsqueeze(int axis){
            return new Tensor<T>(this.TMemory, this.TLayout.AddAxis(axis, 1, 0));
        }
    }
}