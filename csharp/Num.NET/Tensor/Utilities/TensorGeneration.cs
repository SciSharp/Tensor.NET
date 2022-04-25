using System.Buffers;
using System.Runtime.InteropServices;
using Numnet.Native;
using Numnet.Common;

namespace Numnet{
    public partial class Tensor
    {
        public static Tensor<T> Zeros<T>(int[] shape)where T:struct{
            Tensor<T> res = new Tensor<T>(new TensorLayout(shape, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype));
            res.TMemory.AsByteSpan().Fill(0);
            return res;
        }
        public static Tensor<T> FromSpan<T>(Span<T> data, TensorShape shape)where T:struct{
            if(shape.TotalElemCount() != data.Length){
                // TODO
                throw new Exception();
            }
            TensorMemory<T> memory = new TensorMemory<T>(data);
            return new Tensor<T>(memory, new TensorLayout(shape, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype));
        }
        public static Tensor<T> FromArray<T>(T[] data, TensorShape shape)where T:struct{
            if(shape.TotalElemCount() != data.Length){
                // TODO
                throw new Exception();
            }
            TensorMemory<T> memory = new TensorMemory<T>(data);
            return new Tensor<T>(memory, new TensorLayout(shape, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype));
        }
        public static Tensor<T> FromArray<T>(T[] data, int[] shape)where T:struct{
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