
namespace Tensornet{
    public partial class Tensor<T>{
        /// <summary>
        /// Expand the shape of a tensor. Insert a new axis that will appear at the axis position in the expanded array shape.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html?highlight=expand_dims#numpy.expand_dims
        /// </summary>
        /// <param name="axis"> Position in the expanded axes where the new axis (or axes) is placed. </param>
        /// <returns></returns>
        public Tensor<T> Unsqueeze(int axis){
            return new Tensor<T>(this.TMemory, this.TLayout.AddAxis(axis, 1, 0));
        }
    }
}