using System.Buffers;
using System.Runtime.InteropServices;
using Numnet.Native;
using Numnet.Common;

namespace Numnet{
    public partial class Tensor
    {
        public static Tensor<T> FromSpan<T>(Span<T> data, TensorShape shape) where T : struct, IEquatable<T>, IConvertible{
            if(shape.TotalElemCount() != data.Length){
                // TODO
                throw new Exception();
            }
            TensorMemory<T> memory = new TensorMemory<T>(data);
            return new Tensor<T>(memory, new TensorLayout(shape, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype));
        }
        public static Tensor<T> FromArray<T>(T[] data, TensorShape shape) where T : struct, IEquatable<T>, IConvertible{
            if(shape.TotalElemCount() != data.Length){
                // TODO
                throw new Exception();
            }
            TensorMemory<T> memory = new TensorMemory<T>(data);
            return new Tensor<T>(memory, new TensorLayout(shape, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype));
        }
        public static Tensor<T> FromArray<T>(T[] data, int[] shape) where T : struct, IEquatable<T>, IConvertible{
            TensorLayout layout = new TensorLayout(shape, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
            if(layout.TotalElemCount() != data.Length){
                // TODO
                throw new Exception();
            }
            TensorMemory<T> memory = new TensorMemory<T>(data);
            return new Tensor<T>(memory, layout);
        }
    }
}