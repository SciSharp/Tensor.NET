using System.Buffers;
using System.Runtime.InteropServices;
using Tensornet.Native;
using Tensornet.Common;

namespace Tensornet{
    public partial class Tensor
    {
        /// <summary>
        /// Generate a tensor from a span.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="data"> The span which contains the data. </param>
        /// <param name="shape"> The target shape. </param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static Tensor<T> FromSpan<T>(Span<T> data, TensorShape shape) where T : struct, IEquatable<T>, IConvertible{
            if(shape.TotalElemCount() != data.Length){
                // TODO
                throw new Exception();
            }
            TensorMemory<T> memory = new TensorMemory<T>(data);
            return new Tensor<T>(memory, new TensorLayout(shape, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype));
        }
        /// <summary>
        /// Generate a tensor from an array.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="data"> The array which contains the data. </param>
        /// <param name="shape"> The target shape. </param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static Tensor<T> FromArray<T>(T[] data, TensorShape shape) where T : struct, IEquatable<T>, IConvertible{
            if(shape.TotalElemCount() != data.Length){
                // TODO
                throw new Exception();
            }
            TensorMemory<T> memory = new TensorMemory<T>(data);
            return new Tensor<T>(memory, new TensorLayout(shape, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype));
        }
        /// <summary>
        /// Generate a tensor from an array.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="data"> The array which contains the data. </param>
        /// <param name="shape"> The target shape. </param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static Tensor<T> FromArray<T>(T[] data, int[] shape) where T : struct, IEquatable<T>, IConvertible{
            TensorLayout layout = new TensorLayout(shape, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
            if(layout.TotalElemCount() != data.Length){
                // TODO
                throw new Exception();
            }
            TensorMemory<T> memory = new TensorMemory<T>(data);
            return new Tensor<T>(memory, layout);
        }
        /// <summary>
        /// Generate a tensor from an IEnumerable.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="data"> The IEnumerable which contains the data. </param>
        /// <param name="shape"> The target shape. </param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static Tensor<T> FromEnumerable<T>(IEnumerable<T> data, int[] shape) where T : struct, IEquatable<T>, IConvertible{
            return FromArray(data.ToArray(), shape);
        }
    }
}