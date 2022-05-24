using Tensornet.Common;

namespace Tensornet{
    public partial class Tensor<T>{
        /// <summary>
        /// Reshape the tensor to the given shape.
        /// Please note that the returned tensor shares the memory with the source tensor.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.reshape.html?highlight=reshape#numpy.reshape
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor<T> Reshape(Span<int> shape){
            return new Tensor<T>(this.TMemory, this.TLayout.Reshape(new TensorShape(shape), false));
        }
        /// <summary>
        /// Reshape the tensor to the given shape.
        /// Please note that the returned tensor shares the memory with the source tensor.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.reshape.html?highlight=reshape#numpy.reshape
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor<T> Reshape(int[] shape){
            return new Tensor<T>(this.TMemory, this.TLayout.Reshape(new TensorShape(shape), false));
        }
        /// <summary>
        /// Reshape the tensor to the given shape.
        /// Please note that the returned tensor shares the memory with the source tensor.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.reshape.html?highlight=reshape#numpy.reshape
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor<T> Reshape(TensorShape shape){
            return new Tensor<T>(this.TMemory, this.TLayout.Reshape(shape, false));
        }

        /// <summary>
        /// Reshape the tensor to the given shape in place.
        /// Please note that the returned tensor shares the memory with the source tensor.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.reshape.html?highlight=reshape#numpy.reshape
        /// </summary>
        /// <param name="shape"></param>
        public void ReshapeTo(int[] shape){
            TLayout = TLayout.Reshape(new TensorShape(shape), false);
        }
        /// <summary>
        /// Reshape the tensor to the given shape in place.
        /// Please note that the returned tensor shares the memory with the source tensor.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.reshape.html?highlight=reshape#numpy.reshape
        /// </summary>
        /// <param name="shape"></param>
        public void ReshapeTo(TensorShape shape){
            TLayout = TLayout.Reshape(shape, false);
        }
    }
}