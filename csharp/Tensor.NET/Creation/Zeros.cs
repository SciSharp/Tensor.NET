using Tensornet.Common;
using Tensornet.Native.Param;
using Tensornet.Native;

namespace Tensornet{
    public static partial class Tensor{
        /// <summary>
        /// Return a new tensor of given shape and type, filled with zeros.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.zeros.html#numpy.zeros
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="shape"> Shape of the new tensor. </param>
        /// <returns></returns>
        public static Tensor<T> Zeros<T>(TensorShape shape) where T : struct, IConvertible, IEquatable<T>{
            Tensor<T> res = new Tensor<T>(new TensorLayout(shape, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype));
            ZerosInternal(res);
            return res;
        }
        private unsafe static void ZerosInternal<T>(Tensor<T> src) where T : struct, IEquatable<T>, IConvertible{
            FillParam param = new FillParam() { value = .0 };
            IntPtr status = NativeExecutor.Execute(NativeApi.Fill, src.TMemory, src.TLayout, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
        /// <summary>
        /// Return an array of zeros with the same shape and type as a given array.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html#numpy.zeros_like
        /// </summary>
        /// <typeparam name="TResult"></typeparam>
        /// <typeparam name="TRefer"></typeparam>
        /// <param name="tensor"> The tensor with target shape. </param>
        /// <returns></returns>

        public static Tensor<TResult> ZerosLike<TResult, TRefer>(Tensor<TRefer> tensor) 
            where TResult : struct, IConvertible, IEquatable<TResult> 
            where TRefer : struct, IConvertible, IEquatable<TRefer>{
            return Tensor.Zeros<TResult>(tensor.TLayout);
        }
    }
}