using System.Buffers;
using System.Runtime.InteropServices;
using Numnet.Native;
using Numnet.Common;

namespace Numnet{
    public partial class Tensor
    {
        public static Tensor Zeros(int[] shape, DType dtype){
            Tensor res = new Tensor(new TensorLayout(dtype, shape));
            res.TMemory.AsSpan().Fill(0);
            return res;
        }
        public static Tensor<T> Zeros<T>(int[] shape)where T:struct{
            Tensor res = new Tensor(new TensorLayout(TensorTypeInfo.GetTypeInfo(typeof(T))._dtype, shape));
            res.TMemory.AsSpan().Fill(0);
            return res.To<T>();
        }
        public static Tensor<T> FromSpan<T>(Span<T> data, TensorShape shape)where T:struct{
            if(shape.TotalElemCount() != data.Length){
                // TODO
                throw new Exception();
            }
            TensorMemory<T> memory = new TensorMemory<T>(data);
            return new Tensor<T>(memory, new TensorLayout(TensorTypeInfo.GetTypeInfo(typeof(T))._dtype, shape));
        }
        public static Tensor<T> FromArray<T>(T[] data, TensorShape shape)where T:struct{
            if(shape.TotalElemCount() != data.Length){
                // TODO
                throw new Exception();
            }
            TensorMemory<T> memory = new TensorMemory<T>(data);
            return new Tensor<T>(memory, new TensorLayout(TensorTypeInfo.GetTypeInfo(typeof(T))._dtype, shape));
        }
        public static Tensor<T> FromArray<T>(T[] data, int[] shape)where T:struct{
            TensorLayout layout = new TensorLayout(TensorTypeInfo.GetTypeInfo(typeof(T))._dtype, shape);
            if(layout.TotalElemCount() != data.Length){
                // TODO
                throw new Exception();
            }
            TensorMemory<T> memory = new TensorMemory<T>(data);
            return new Tensor<T>(memory, layout);
        }
    }
}