using Numnet.Exceptions;

namespace Numnet{
    public partial class Tensor<T>{
        /// <summary>
        /// Squeeze the tensor to remove the dim which equals to one.
        /// </summary>
        /// <param name="axis"> The axis to be removed. If it's set to -1, then all dims of length one will be removed. </param>
        /// <returns></returns>
        public Tensor<T> Squeeze(int axis = -1){
            if(axis < -1){
                throw new InvalidArgumentException("The axis to be squeezed should not be smaller than -1.");
            }
            if(axis >= TLayout.NDim){
                throw new InvalidArgumentException("The axis to be squeezed is too large.");
            }
            var layout = new TensorLayout(this.TLayout);
            if(axis == -1){
                for (int i = TLayout.NDim - 1; i >= 0 && layout.NDim >= 2; i--){
                    if(layout.Shape[i] == 1) layout.RemoveAxisInplace(i);
                }
            }
            else if(TLayout.Shape[axis] == 1){
                layout.RemoveAxisInplace(axis);
            }
            else{
                throw new InvalidArgumentException("The axis to be squeezed does not have value 1.");
            }
            return new Tensor<T>(this.TMemory, layout);
        }
    }
}