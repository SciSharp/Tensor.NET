using System.IO;
using System.Text;
using Numnet.Common;
using Numnet.Native;
using Numnet.Exceptions;

namespace Numnet{
    public enum TensorSerializationMode
    {
        Auto = 0,
        NumNET,
        Numpy
    }
    public static class TensorWriter{
        private static byte[] _magicNumber;
        private static byte[] _versionNumber;

        static TensorWriter(){
            _magicNumber = Encoding.ASCII.GetBytes("NumNET");
            _versionNumber = Encoding.ASCII.GetBytes(Tensor.VersionNumber);
        }

        public static void Write<T>(string path, Tensor<T> tensor, TensorSerializationMode mode = TensorSerializationMode.NumNET) where T : struct, IEquatable<T>, IConvertible{
            using(var fs = new FileStream(path, FileMode.OpenOrCreate, FileAccess.Write)){
                WriteHeader(fs, tensor.TLayout);
                if(tensor.TLayout.IsContiguous()){
                    WriteContiguousData(fs, tensor.TMemory);
                }
                else{
                    WriteStrideData(fs, tensor);
                }
            }
        }

        public static async void WriteAsync<T>(string path, Tensor<T> tensor) where T : struct, IEquatable<T>, IConvertible{
            
        }

        private static void WriteMagicAndVersion(FileStream fs){
            fs.Write(_magicNumber);
            fs.Write(_versionNumber);
        }

        private static void WriteHeader(FileStream fs, TensorLayout layout){
            WriteMagicAndVersion(fs);
            fs.Write(BitConverter.GetBytes(((int)layout.DType)));
            fs.Write(BitConverter.GetBytes(layout.NDim));
            for (int i = 0; i < TensorLayout.MAX_NDIM; i++)
            {
                fs.Write(BitConverter.GetBytes(layout.Shape[i]));
            }
        }

        private static void WriteContiguousData<T>(FileStream fs, TensorMemory<T> memory) where T : struct, IEquatable<T>, IConvertible{
            fs.Write(memory.AsByteSpan());
        }

        private static void WriteStrideData<T>(FileStream fs, Tensor<T> tensor) where T : struct, IEquatable<T>, IConvertible{
            byte[] data = new byte[TensorTypeInfo.GetTypeSize(tensor.TLayout.DType)];
            Span<byte> dataSpan = new Span<byte>(data);
            foreach(T item in tensor){
                bool actionRes = tensor.TLayout.DType switch
                {
                    DType.Int32 => BitConverter.TryWriteBytes(dataSpan, Convert.ToInt32(item)),
                    DType.Int64 => BitConverter.TryWriteBytes(dataSpan, Convert.ToInt64(item)),
                    DType.Float32 => BitConverter.TryWriteBytes(dataSpan, Convert.ToSingle(item)),
                    DType.Float64 => BitConverter.TryWriteBytes(dataSpan, Convert.ToDouble(item)),
                    DType.Bool => BitConverter.TryWriteBytes(dataSpan, Convert.ToBoolean(item)),
                    _ => false
                };
                if(!actionRes){
                    throw new NNRuntimeException("Failed to write byte to target span.");
                }
                fs.Write(dataSpan);
            }
        }
    }
}